#!/usr/bin/env python3
"""Report cumulative and latest-candidate CI work for a pull-request cohort.

The report deliberately separates raw summed hosted job execution, workflow
wall time, and agent waiting. GitHub supplies the first two. Agent waiting and
semantic causes that cannot be derived safely come from an attribution ledger.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


GITHUB_SCHEMA = "chelis-ci-pr-lifecycle-github-v1"
LEDGER_SCHEMA = "chelis-ci-pr-lifecycle-attribution-v1"
REPORT_SCHEMA = "chelis-ci-pr-lifecycle-report-v1"

CI_WORKFLOWS = {
    ".github/workflows/ci.yml": "CI",
    ".github/workflows/conformance.yml": "Hull",
}
EXPANSION_WORKFLOW = ".github/workflows/pr-package-expansion.yml"
CAUSES = (
    "initial-candidate",
    "review-repair",
    "ordinary-content-push",
    "non-conflicting-rebase-base-update",
    "trivial-or-hand-resolved-conflict-rebase",
    "base-retarget-stack-collapse",
    "ci-policy-ci-repair",
    "package-expansion-rerun",
    "unknown",
)


class ReportError(ValueError):
    """The requested report cannot be produced from the supplied evidence."""


Api = Callable[[str], object]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ReportError(f"{label} must be an array")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ReportError(f"{label} must be a positive integer")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReportError(f"{label} must be nonempty text")
    return value.strip()


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _timestamp(value: object, label: str) -> datetime:
    text = _text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ReportError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ReportError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _duration_seconds(start: object, end: object, label: str) -> float:
    started = _timestamp(start, f"{label} start")
    completed = _timestamp(end, f"{label} end")
    seconds = (completed - started).total_seconds()
    if seconds < 0:
        raise ReportError(f"{label} ends before it starts")
    return seconds


def _minutes(seconds: float) -> float:
    return round(seconds / 60.0, 3)


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 3)


def _workflow_path(run: Mapping[str, Any]) -> str:
    value = run.get("workflow_path", run.get("path"))
    path = _text(value, "workflow run path")
    return path.split("@", 1)[0]


def _head_sha(run: Mapping[str, Any]) -> str:
    return _text(run.get("head_sha"), "workflow run head_sha")


def _candidate_id(run: Mapping[str, Any]) -> str:
    value = run.get("candidate_sha", run.get("candidate_id"))
    if value is None:
        return _head_sha(run)
    return _text(value, "workflow run candidate_sha")


def _run_time(run: Mapping[str, Any]) -> datetime:
    for key in ("run_started_at", "created_at", "updated_at"):
        if run.get(key) is not None:
            return _timestamp(run[key], f"workflow run {key}")
    raise ReportError("workflow run requires a stable timestamp")


def _job_seconds(run: Mapping[str, Any]) -> float:
    total = 0.0
    for index, raw_job in enumerate(
        _sequence(run.get("jobs", []), "workflow run jobs")
    ):
        job = _mapping(raw_job, f"workflow run job {index}")
        if (
            job.get("conclusion") == "skipped"
            or job.get("started_at") is None
            or job.get("completed_at") is None
        ):
            continue
        started = _timestamp(job["started_at"], "workflow job started_at")
        completed = _timestamp(job["completed_at"], "workflow job completed_at")
        if completed >= started:
            total += (completed - started).total_seconds()
    return total


def _untimed_job_count(run: Mapping[str, Any]) -> int:
    count = 0
    for index, raw_job in enumerate(
        _sequence(run.get("jobs", []), "workflow run jobs")
    ):
        job = _mapping(raw_job, f"workflow run job {index}")
        if job.get("conclusion") == "skipped":
            continue
        if job.get("started_at") is None or job.get("completed_at") is None:
            count += 1
    return count


def _timing_anomaly_count(run: Mapping[str, Any]) -> int:
    count = 0
    for index, raw_job in enumerate(
        _sequence(run.get("jobs", []), "workflow run jobs")
    ):
        job = _mapping(raw_job, f"workflow run job {index}")
        if (
            job.get("conclusion") == "skipped"
            or job.get("started_at") is None
            or job.get("completed_at") is None
        ):
            continue
        started = _timestamp(job["started_at"], "workflow job started_at")
        completed = _timestamp(job["completed_at"], "workflow job completed_at")
        if completed < started:
            count += 1
    return count


def _attempt_windows(run: Mapping[str, Any]) -> dict[int, tuple[datetime, datetime]]:
    windows: dict[int, tuple[datetime, datetime]] = {}
    for index, raw_job in enumerate(
        _sequence(run.get("jobs", []), "workflow run jobs")
    ):
        job = _mapping(raw_job, f"workflow run job {index}")
        if (
            job.get("conclusion") == "skipped"
            or job.get("started_at") is None
            or job.get("completed_at") is None
        ):
            continue
        attempt_value = job.get("run_attempt", run.get("run_attempt", 1))
        attempt = _positive_int(attempt_value, "workflow job run_attempt")
        started = _timestamp(job["started_at"], "workflow job started_at")
        completed = _timestamp(job["completed_at"], "workflow job completed_at")
        if completed < started:
            continue
        prior = windows.get(attempt)
        if prior is None:
            windows[attempt] = (started, completed)
        else:
            windows[attempt] = (min(prior[0], started), max(prior[1], completed))
    if windows:
        return windows
    if run.get("run_started_at") is not None and run.get("updated_at") is not None:
        attempt = _positive_int(
            run.get("run_attempt", 1), "workflow run_attempt"
        )
        started = _timestamp(run["run_started_at"], "workflow run_started_at")
        completed = _timestamp(run["updated_at"], "workflow updated_at")
        if completed < started:
            raise ReportError("workflow run updated before it started")
        windows[attempt] = (started, completed)
    return windows


def _workflow_wall_seconds(run: Mapping[str, Any]) -> float:
    return sum(
        (completed - started).total_seconds()
        for started, completed in _attempt_windows(run).values()
    )


def _attempt_count(run: Mapping[str, Any]) -> int:
    windows = _attempt_windows(run)
    if windows:
        return len(windows)
    return 1


def parse_prs(specification: str) -> list[int]:
    numbers: set[int] = set()
    for raw_part in specification.split(","):
        part = raw_part.strip()
        if not part:
            raise ReportError("PR cohort contains an empty item")
        try:
            if "-" not in part:
                numbers.add(_positive_int(int(part), "PR number"))
                continue
            fields = part.split("-")
            if len(fields) != 2:
                raise ReportError(f"invalid PR range {part!r}")
            start = _positive_int(int(fields[0]), "PR range start")
            end = _positive_int(int(fields[1]), "PR range end")
        except ValueError as error:
            raise ReportError(f"invalid PR cohort item {part!r}") from error
        if end < start:
            raise ReportError(f"descending PR range {part!r}")
        numbers.update(range(start, end + 1))
    if not numbers:
        raise ReportError("PR cohort must not be empty")
    return sorted(numbers)


def _validate_cause(value: object, label: str) -> str:
    cause = _text(value, label)
    if cause not in CAUSES:
        raise ReportError(
            f"{label} must be one of {', '.join(CAUSES)}; got {cause!r}"
        )
    return cause


def _empty_ledger() -> dict[str, object]:
    return {
        "schema": LEDGER_SCHEMA,
        "manual_run_scope_complete": False,
        "manual_run_scope_evidence": None,
        "candidate_attributions": [],
        "run_attributions": [],
        "agent_waits": [],
    }


def validate_ledger(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != LEDGER_SCHEMA:
        raise ReportError(f"attribution ledger schema must be {LEDGER_SCHEMA}")
    manual_run_scope_complete = payload.get("manual_run_scope_complete", False)
    if type(manual_run_scope_complete) is not bool:
        raise ReportError("manual_run_scope_complete must be a boolean")
    manual_run_scope_evidence = _optional_text(
        payload.get("manual_run_scope_evidence"),
        "manual_run_scope_evidence",
    )
    if manual_run_scope_complete and manual_run_scope_evidence is None:
        raise ReportError(
            "manual_run_scope_evidence is required when manual run scope is complete"
        )

    candidate_attributions: list[dict[str, Any]] = []
    candidate_keys: set[tuple[int, str, str | None]] = set()
    for index, raw in enumerate(
        _sequence(
            payload.get("candidate_attributions", []),
            "candidate_attributions",
        )
    ):
        row = _mapping(raw, f"candidate attribution {index}")
        pr_number = _positive_int(row.get("pr_number"), "candidate PR number")
        head_sha = _text(row.get("head_sha"), "candidate head_sha")
        candidate_sha = _optional_text(
            row.get("candidate_sha"), "candidate candidate_sha"
        )
        key = (pr_number, head_sha, candidate_sha)
        if key in candidate_keys:
            raise ReportError(f"duplicate candidate attribution for {key}")
        candidate_keys.add(key)
        candidate_attributions.append(
            {
                "pr_number": pr_number,
                "head_sha": head_sha,
                "candidate_sha": candidate_sha,
                "cause": _validate_cause(row.get("cause"), "candidate cause"),
                "evidence": _text(
                    row.get("evidence"), "candidate attribution evidence"
                ),
                "note": _optional_text(row.get("note"), "candidate note"),
            }
        )

    run_attributions: list[dict[str, Any]] = []
    run_ids: set[int] = set()
    for index, raw in enumerate(
        _sequence(payload.get("run_attributions", []), "run_attributions")
    ):
        row = _mapping(raw, f"run attribution {index}")
        run_id = _positive_int(row.get("run_id"), "attributed run id")
        if run_id in run_ids:
            raise ReportError(f"duplicate run attribution for {run_id}")
        run_ids.add(run_id)
        run_attributions.append(
            {
                "run_id": run_id,
                "pr_number": _positive_int(
                    row.get("pr_number"), "attributed run PR number"
                ),
                "head_sha": _text(
                    row.get("head_sha"), "attributed run head_sha"
                ),
                "candidate_sha": _optional_text(
                    row.get("candidate_sha"), "attributed run candidate_sha"
                ),
                "cause": _validate_cause(row.get("cause"), "run cause"),
                "evidence": _text(
                    row.get("evidence"), "run attribution evidence"
                ),
                "note": _optional_text(row.get("note"), "run note"),
            }
        )

    agent_waits: list[dict[str, Any]] = []
    wait_ids: set[str] = set()
    for index, raw in enumerate(
        _sequence(payload.get("agent_waits", []), "agent_waits")
    ):
        row = _mapping(raw, f"agent wait {index}")
        wait_id = _text(row.get("wait_id"), "agent wait_id")
        if wait_id in wait_ids:
            raise ReportError(f"duplicate agent wait_id {wait_id!r}")
        wait_ids.add(wait_id)
        started_at = _timestamp(row.get("started_at"), "agent wait started_at")
        ended_at = _timestamp(row.get("ended_at"), "agent wait ended_at")
        if ended_at < started_at:
            raise ReportError(f"agent wait {wait_id!r} ends before it starts")
        run_ids_value = [
            _positive_int(run_id, f"agent wait {wait_id} run id")
            for run_id in _sequence(
                row.get("run_ids", []), f"agent wait {wait_id} run_ids"
            )
        ]
        agent_waits.append(
            {
                "wait_id": wait_id,
                "pr_number": _positive_int(
                    row.get("pr_number"), "agent wait PR number"
                ),
                "head_sha": _optional_text(
                    row.get("head_sha"), "agent wait head_sha"
                ),
                "cause": _validate_cause(row.get("cause"), "agent wait cause"),
                "started_at": _timestamp_text(started_at),
                "ended_at": _timestamp_text(ended_at),
                "run_ids": run_ids_value,
                "evidence": _text(row.get("evidence"), "agent wait evidence"),
                "note": _optional_text(row.get("note"), "agent wait note"),
            }
        )

    return {
        "schema": LEDGER_SCHEMA,
        "manual_run_scope_complete": manual_run_scope_complete,
        "manual_run_scope_evidence": manual_run_scope_evidence,
        "candidate_attributions": candidate_attributions,
        "run_attributions": run_attributions,
        "agent_waits": agent_waits,
    }


def _candidate_attribution(
    *,
    pr_number: int,
    head_sha: str,
    candidate_sha: str,
    ledger: Mapping[str, Any],
) -> dict[str, str | None]:
    matches = [
        row
        for row in ledger["candidate_attributions"]
        if row["pr_number"] == pr_number
        and row["head_sha"] == head_sha
        and (
            row["candidate_sha"] is None
            or row["candidate_sha"] == candidate_sha
        )
    ]
    if len(matches) > 1:
        raise ReportError(
            f"multiple ledger rows match PR #{pr_number} candidate {candidate_sha}"
        )
    if matches:
        row = matches[0]
        return {
            "cause": row["cause"],
            "attribution": "ledger",
            "evidence": row["evidence"],
            "note": row["note"],
        }
    return {
        "cause": "unknown",
        "attribution": "unknown",
        "evidence": "GitHub timing and head identity do not establish push cause",
        "note": None,
    }


def _run_attribution(
    run: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any] | None,
    expansion_ordinal: int | None,
    ledger: Mapping[str, Any],
) -> dict[str, str | None]:
    run_id = _positive_int(run.get("id"), "workflow run id")
    matches = [
        row for row in ledger["run_attributions"] if row["run_id"] == run_id
    ]
    if matches:
        row = matches[0]
        return {
            "cause": row["cause"],
            "attribution": "ledger",
            "evidence": row["evidence"],
            "note": row["note"],
        }
    if candidate is not None:
        return {
            "cause": candidate["cause"],
            "attribution": candidate["attribution"],
            "evidence": candidate["evidence"],
            "note": candidate["note"],
        }
    if expansion_ordinal is not None and expansion_ordinal > 1:
        return {
            "cause": "package-expansion-rerun",
            "attribution": "inferred",
            "evidence": "later package-expansion run observed for the same PR",
            "note": None,
        }
    return {
        "cause": "unknown",
        "attribution": "unknown",
        "evidence": "no supplied ledger row establishes this run's cause",
        "note": None,
    }


def _run_pr_number(
    run: Mapping[str, Any], run_attributions: Mapping[int, Mapping[str, Any]]
) -> int | None:
    value = run.get("pr_number")
    if type(value) is int and value > 0:
        return value
    run_id = _positive_int(run.get("id"), "workflow run id")
    attribution = run_attributions.get(run_id)
    if attribution is not None:
        return attribution["pr_number"]
    return None


def _normalized_run(
    raw: Mapping[str, Any],
    run_attributions: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    run = dict(raw)
    run_id = _positive_int(run.get("id"), "workflow run id")
    attribution = run_attributions.get(run_id)
    supplied_pr_number = run.get("pr_number")
    supplied_head_sha = run.get("head_sha")
    if (
        attribution is not None
        and type(supplied_pr_number) is int
        and supplied_pr_number != attribution["pr_number"]
    ):
        raise ReportError(
            f"run {run_id} PR attribution disagrees with supplied GitHub data"
        )
    if (
        attribution is not None
        and run.get("event") == "pull_request"
        and isinstance(supplied_head_sha, str)
        and supplied_head_sha != attribution["head_sha"]
    ):
        raise ReportError(
            f"run {run_id} head attribution disagrees with supplied GitHub data"
        )
    pr_number = _run_pr_number(run, run_attributions)
    if pr_number is not None:
        run["pr_number"] = pr_number
    if attribution is not None:
        run["head_sha"] = attribution["head_sha"]
        if attribution["candidate_sha"] is not None:
            run["candidate_sha"] = attribution["candidate_sha"]
    _workflow_path(run)
    _run_time(run)
    _sequence(run.get("jobs", []), "workflow run jobs")
    return run


def _pull_requests(
    github_payload: Mapping[str, Any], cohort: Sequence[int]
) -> dict[int, dict[str, Any]]:
    pulls: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(
        _sequence(github_payload.get("pull_requests"), "pull_requests")
    ):
        pull = _mapping(raw, f"pull request {index}")
        number = _positive_int(pull.get("number"), "pull request number")
        if number in pulls:
            raise ReportError(f"duplicate pull request metadata for #{number}")
        pulls[number] = {
            "number": number,
            "title": _optional_text(pull.get("title"), "pull request title"),
            "state": _optional_text(pull.get("state"), "pull request state"),
            "created_at": _timestamp_text(
                _timestamp(pull.get("created_at"), "pull request created_at")
            ),
            "updated_at": (
                _timestamp_text(
                    _timestamp(pull.get("updated_at"), "pull request updated_at")
                )
                if pull.get("updated_at") is not None
                else None
            ),
            "head_sha": _text(pull.get("head_sha"), "pull request head_sha"),
            "head_ref": _optional_text(
                pull.get("head_ref"), "pull request head_ref"
            ),
            "head_repository": _optional_text(
                pull.get("head_repository"), "pull request head_repository"
            ),
            "base_ref": _optional_text(
                pull.get("base_ref"), "pull request base_ref"
            ),
        }
    missing = sorted(set(cohort).difference(pulls))
    if missing:
        raise ReportError(f"GitHub input is missing pull requests {missing}")
    return pulls


def build_report(
    github_payload: Mapping[str, Any],
    *,
    prs: Sequence[int],
    attribution_payload: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    if github_payload.get("schema") != GITHUB_SCHEMA:
        raise ReportError(f"GitHub input schema must be {GITHUB_SCHEMA}")
    repository = _text(github_payload.get("repository"), "repository")
    cohort = sorted({_positive_int(value, "PR number") for value in prs})
    if not cohort:
        raise ReportError("PR cohort must not be empty")
    pull_requests = _pull_requests(github_payload, cohort)
    ledger = validate_ledger(
        attribution_payload if attribution_payload is not None else _empty_ledger()
    )
    run_attributions = {
        row["run_id"]: row for row in ledger["run_attributions"]
    }
    runs = [
        _normalized_run(
            _mapping(raw, f"workflow run {index}"), run_attributions
        )
        for index, raw in enumerate(
            _sequence(github_payload.get("workflow_runs"), "workflow_runs")
        )
    ]
    runs = [
        run
        for run in runs
        if run.get("pr_number") in cohort
        and (
            _workflow_path(run) in CI_WORKFLOWS
            or _workflow_path(run) == EXPANSION_WORKFLOW
        )
    ]

    implementation_runs = [
        run for run in runs if _workflow_path(run) in CI_WORKFLOWS
    ]
    expansion_runs = [
        run for run in runs if _workflow_path(run) == EXPANSION_WORKFLOW
    ]

    candidate_groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for run in implementation_runs:
        pr_number = _positive_int(run.get("pr_number"), "workflow run PR number")
        candidate_groups[(pr_number, _candidate_id(run))].append(run)

    candidate_rows: list[dict[str, Any]] = []
    candidate_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for pr_number in cohort:
        keys = [
            key for key in candidate_groups if key[0] == pr_number
        ]
        keys.sort(
            key=lambda key: min(_run_time(run) for run in candidate_groups[key])
        )
        for ordinal, key in enumerate(keys, start=1):
            grouped_runs = candidate_groups[key]
            head_shas = {_head_sha(run) for run in grouped_runs}
            if len(head_shas) != 1:
                raise ReportError(
                    f"PR #{pr_number} candidate {key[1]} has multiple head SHAs"
                )
            head_sha = head_shas.pop()
            attribution = _candidate_attribution(
                pr_number=pr_number,
                head_sha=head_sha,
                candidate_sha=key[1],
                ledger=ledger,
            )
            job_seconds = sum(_job_seconds(run) for run in grouped_runs)
            wall_seconds = sum(
                _workflow_wall_seconds(run) for run in grouped_runs
            )
            row = {
                "pr_number": pr_number,
                "ordinal": ordinal,
                "head_sha": head_sha,
                "candidate_sha": key[1],
                "first_observed_at": _timestamp_text(
                    min(_run_time(run) for run in grouped_runs)
                ),
                "cause": attribution["cause"],
                "attribution": attribution["attribution"],
                "evidence": attribution["evidence"],
                "note": attribution["note"],
                "workflow_run_ids": sorted(
                    _positive_int(run.get("id"), "workflow run id")
                    for run in grouped_runs
                ),
                "workflow_run_count": len(grouped_runs),
                "workflow_attempt_count": sum(
                    _attempt_count(run) for run in grouped_runs
                ),
                "raw_job_minutes": _minutes(job_seconds),
                "workflow_wall_minutes": _minutes(wall_seconds),
            }
            candidate_rows.append(row)
            candidate_by_key[key] = row

    expansion_by_pr: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for run in expansion_runs:
        pr_number = _positive_int(run.get("pr_number"), "expansion run PR number")
        expansion_by_pr[pr_number].append(run)
    for grouped in expansion_by_pr.values():
        grouped.sort(key=_run_time)

    run_rows: list[dict[str, Any]] = []
    for run in sorted(runs, key=_run_time):
        run_id = _positive_int(run.get("id"), "workflow run id")
        pr_number = _positive_int(run.get("pr_number"), "workflow run PR number")
        path = _workflow_path(run)
        candidate = (
            candidate_by_key.get((pr_number, _candidate_id(run)))
            if path in CI_WORKFLOWS
            else None
        )
        expansion_ordinal = None
        if path == EXPANSION_WORKFLOW:
            expansion_ordinal = (
                expansion_by_pr[pr_number].index(run) + 1
            )
        attribution = _run_attribution(
            run,
            candidate=candidate,
            expansion_ordinal=expansion_ordinal,
            ledger=ledger,
        )
        pr_attribution = _optional_text(
            run.get("pr_attribution"), "workflow run PR attribution"
        )
        run_rows.append(
            {
                "run_id": run_id,
                "pr_number": pr_number,
                "pr_attribution": (
                    pr_attribution
                    if pr_attribution is not None
                    else "provided GitHub input"
                ),
                "head_sha": _head_sha(run),
                "candidate_sha": (
                    _candidate_id(run) if path in CI_WORKFLOWS else None
                ),
                "workflow": (
                    CI_WORKFLOWS[path]
                    if path in CI_WORKFLOWS
                    else "Package Expansion"
                ),
                "workflow_path": path,
                "event": _optional_text(run.get("event"), "workflow run event"),
                "started_at": _timestamp_text(_run_time(run)),
                "cause": attribution["cause"],
                "attribution": attribution["attribution"],
                "evidence": attribution["evidence"],
                "note": attribution["note"],
                "attempt_count": _attempt_count(run),
                "raw_job_minutes": _minutes(_job_seconds(run)),
                "workflow_wall_minutes": _minutes(
                    _workflow_wall_seconds(run)
                ),
                "untimed_job_count": _untimed_job_count(run),
                "timing_anomaly_count": _timing_anomaly_count(run),
            }
        )

    wait_rows: list[dict[str, Any]] = []
    for row in ledger["agent_waits"]:
        if row["pr_number"] not in cohort:
            continue
        wait_seconds = _duration_seconds(
            row["started_at"],
            row["ended_at"],
            f"agent wait {row['wait_id']}",
        )
        wait_rows.append({**row, "wait_minutes": _minutes(wait_seconds)})

    cause_rows: list[dict[str, Any]] = []
    for cause in CAUSES:
        cause_candidates = [row for row in candidate_rows if row["cause"] == cause]
        cause_runs = [row for row in run_rows if row["cause"] == cause]
        cause_waits = [row for row in wait_rows if row["cause"] == cause]
        ci_runs = [
            row for row in cause_runs if row["workflow_path"] in CI_WORKFLOWS
        ]
        expansion_cause_runs = [
            row
            for row in cause_runs
            if row["workflow_path"] == EXPANSION_WORKFLOW
        ]
        cause_rows.append(
            {
                "cause": cause,
                "candidate_count": len(cause_candidates),
                "workflow_run_count": len(cause_runs),
                "workflow_attempt_count": sum(
                    row["attempt_count"] for row in cause_runs
                ),
                "ci_hull_raw_job_minutes": round(
                    sum(row["raw_job_minutes"] for row in ci_runs), 3
                ),
                "package_expansion_raw_job_minutes": round(
                    sum(
                        row["raw_job_minutes"]
                        for row in expansion_cause_runs
                    ),
                    3,
                ),
                "all_hosted_raw_job_minutes": round(
                    sum(row["raw_job_minutes"] for row in cause_runs), 3
                ),
                "summed_workflow_wall_minutes": round(
                    sum(row["workflow_wall_minutes"] for row in cause_runs), 3
                ),
                "agent_wait_count": len(cause_waits),
                "agent_wait_minutes": round(
                    sum(row["wait_minutes"] for row in cause_waits), 3
                ),
            }
        )

    pr_rows: list[dict[str, Any]] = []
    for pr_number in cohort:
        pull = pull_requests[pr_number]
        pr_candidates = [
            row for row in candidate_rows if row["pr_number"] == pr_number
        ]
        pr_candidates.sort(key=lambda row: row["ordinal"])
        latest = pr_candidates[-1] if pr_candidates else None
        cumulative = sum(row["raw_job_minutes"] for row in pr_candidates)
        latest_minutes = latest["raw_job_minutes"] if latest else 0.0
        pr_expansions = [
            row
            for row in run_rows
            if row["pr_number"] == pr_number
            and row["workflow_path"] == EXPANSION_WORKFLOW
        ]
        pr_waits = [row for row in wait_rows if row["pr_number"] == pr_number]
        pr_rows.append(
            {
                "pr_number": pr_number,
                "title": pull["title"],
                "state": pull["state"],
                "created_at": pull["created_at"],
                "current_head_sha": pull["head_sha"],
                "base_ref": pull["base_ref"],
                "candidate_count": len(pr_candidates),
                "package_expansion_count": len(pr_expansions),
                "cumulative_ci_hull_raw_job_minutes": round(cumulative, 3),
                "latest_candidate_ci_hull_raw_job_minutes": round(
                    latest_minutes, 3
                ),
                "amplification_ratio": _ratio(cumulative, latest_minutes),
                "package_expansion_raw_job_minutes": round(
                    sum(row["raw_job_minutes"] for row in pr_expansions), 3
                ),
                "agent_wait_minutes": round(
                    sum(row["wait_minutes"] for row in pr_waits), 3
                ),
                "latest_head_sha": latest["head_sha"] if latest else None,
                "latest_candidate_sha": (
                    latest["candidate_sha"] if latest else None
                ),
            }
        )

    cumulative = sum(
        row["cumulative_ci_hull_raw_job_minutes"] for row in pr_rows
    )
    latest = sum(
        row["latest_candidate_ci_hull_raw_job_minutes"] for row in pr_rows
    )
    implementation_run_rows = [
        row for row in run_rows if row["workflow_path"] in CI_WORKFLOWS
    ]
    expansion_run_rows = [
        row for row in run_rows if row["workflow_path"] == EXPANSION_WORKFLOW
    ]
    warnings: list[str] = []
    if not ledger["manual_run_scope_complete"]:
        warnings.append(
            "Manual workflow attribution is not declared complete; retarget "
            "and package-expansion counts are lower bounds until every cohort "
            "dispatch has a run_attributions row."
        )
    collected_since = github_payload.get("collected_since")
    if collected_since is not None:
        since = _timestamp(collected_since, "collected_since")
        truncated = [
            number
            for number in cohort
            if since
            > _timestamp(
                pull_requests[number]["created_at"],
                f"pull request #{number} created_at",
            )
        ]
        if truncated:
            warnings.append(
                "The live collection boundary is later than PR creation for "
                f"{truncated}; candidate history may be incomplete."
            )
    if any(row["attribution"] == "unknown" for row in candidate_rows):
        warnings.append(
            "Some candidate causes are unknown; supply candidate_attributions "
            "to classify them."
        )
    if any(row["attribution"] == "unknown" for row in expansion_run_rows):
        warnings.append(
            "Some package-expansion causes are unknown; supply run_attributions "
            "for semantic classification."
        )
    if not wait_rows:
        warnings.append(
            "No agent wait intervals were supplied; agent waiting is reported "
            "as unavailable rather than inferred from workflow timestamps."
        )
    untimed_jobs = sum(row["untimed_job_count"] for row in run_rows)
    if untimed_jobs:
        warnings.append(
            f"{untimed_jobs} non-skipped jobs lacked complete timestamps and "
            "were excluded from raw job-minute totals."
        )
    timing_anomalies = sum(row["timing_anomaly_count"] for row in run_rows)
    if timing_anomalies:
        warnings.append(
            f"{timing_anomalies} non-skipped jobs had inverted timestamps and "
            "were excluded from raw job-minute and workflow-wall totals."
        )

    generated = (
        _timestamp(generated_at, "generated_at")
        if generated_at is not None
        else datetime.now(timezone.utc)
    )
    return {
        "schema": REPORT_SCHEMA,
        "generated_at": _timestamp_text(generated),
        "repository": repository,
        "cohort": cohort,
        "attribution_scope": {
            "manual_run_scope_complete": ledger["manual_run_scope_complete"],
            "manual_run_scope_evidence": ledger["manual_run_scope_evidence"],
        },
        "definitions": {
            "candidate_count": (
                "Distinct implementation candidate identities observed across "
                "CI and Hull runs; candidate_sha is preferred over head_sha."
            ),
            "raw_job_minutes": (
                "Sum of completed_at-started_at for hosted jobs. This is not "
                "billing-rounded runner time or dollar cost."
            ),
            "workflow_wall_minutes": (
                "Sum of each workflow attempt's earliest-job to latest-job "
                "window. Concurrent attempts and workflows can overlap."
            ),
            "agent_wait_minutes": (
                "Sum of explicit attribution-ledger intervals during which an "
                "agent was waiting for CI. It is never inferred from GitHub."
            ),
            "amplification_ratio": (
                "Cumulative CI/Hull raw job-minutes divided by the sum of each "
                "PR's latest-candidate CI/Hull raw job-minutes."
            ),
        },
        "summary": {
            "candidate_count": len(candidate_rows),
            "implementation_workflow_run_count": len(implementation_run_rows),
            "implementation_workflow_attempt_count": sum(
                row["attempt_count"] for row in implementation_run_rows
            ),
            "package_expansion_count": len(expansion_run_rows),
            "package_expansion_attempt_count": sum(
                row["attempt_count"] for row in expansion_run_rows
            ),
            "cumulative_ci_hull_raw_job_minutes": round(cumulative, 3),
            "latest_candidate_ci_hull_raw_job_minutes": round(latest, 3),
            "amplification_ratio": _ratio(cumulative, latest),
            "package_expansion_raw_job_minutes": round(
                sum(row["raw_job_minutes"] for row in expansion_run_rows), 3
            ),
            "all_measured_hosted_raw_job_minutes": round(
                sum(row["raw_job_minutes"] for row in run_rows), 3
            ),
            "summed_workflow_wall_minutes": round(
                sum(row["workflow_wall_minutes"] for row in run_rows), 3
            ),
            "agent_wait_interval_count": len(wait_rows),
            "agent_wait_minutes": round(
                sum(row["wait_minutes"] for row in wait_rows), 3
            ),
            "untimed_job_count": untimed_jobs,
            "timing_anomaly_count": timing_anomalies,
        },
        "pull_requests": pr_rows,
        "causes": cause_rows,
        "candidate_ledger": candidate_rows,
        "run_ledger": run_rows,
        "agent_wait_ledger": wait_rows,
        "warnings": warnings,
    }


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}".rstrip("0").rstrip(".")


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = _mapping(report.get("summary"), "report summary")
    lines = [
        "# CI candidate lifecycle report",
        "",
        f"- Repository: `{report['repository']}`",
        f"- PR cohort: {', '.join(f'#{number}' for number in report['cohort'])}",
        f"- Generated: {report['generated_at']}",
        "",
        "## Cohort summary",
        "",
        "| Measure | Value |",
        "| --- | ---: |",
        f"| Candidates | {summary['candidate_count']} |",
        (
            "| Package expansions | "
            f"{summary['package_expansion_count']} |"
        ),
        (
            "| Cumulative CI/Hull raw job-minutes | "
            f"{_format_number(summary['cumulative_ci_hull_raw_job_minutes'])} |"
        ),
        (
            "| Latest-candidate CI/Hull raw job-minutes | "
            f"{_format_number(summary['latest_candidate_ci_hull_raw_job_minutes'])} |"
        ),
        (
            "| Lifecycle amplification | "
            f"{_format_number(summary['amplification_ratio'])}x |"
        ),
        (
            "| Package-expansion raw job-minutes | "
            f"{_format_number(summary['package_expansion_raw_job_minutes'])} |"
        ),
        (
            "| All measured hosted raw job-minutes | "
            f"{_format_number(summary['all_measured_hosted_raw_job_minutes'])} |"
        ),
        (
            "| Summed workflow wall-minutes | "
            f"{_format_number(summary['summed_workflow_wall_minutes'])} |"
        ),
        (
            "| Ledger-supplied agent wait minutes | "
            f"{_format_number(summary['agent_wait_minutes'])} |"
        ),
        "",
        "Raw job-minutes sum job execution and are not billing-rounded runner "
        "minutes. Workflow wall time can overlap across workflows. Agent waiting "
        "comes only from supplied intervals and is neither of those measures. "
        "No dollar cost is calculated because runner rates are not part of this "
        "evidence.",
        "",
        "## Pull requests",
        "",
        (
            "| PR | Candidates | Expansions | Cumulative CI/Hull job-min | "
            "Latest job-min | Amplification | Agent wait min |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["pull_requests"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"#{row['pr_number']}",
                    str(row["candidate_count"]),
                    str(row["package_expansion_count"]),
                    _format_number(
                        row["cumulative_ci_hull_raw_job_minutes"]
                    ),
                    _format_number(
                        row["latest_candidate_ci_hull_raw_job_minutes"]
                    ),
                    (
                        f"{_format_number(row['amplification_ratio'])}x"
                        if row["amplification_ratio"] is not None
                        else "n/a"
                    ),
                    _format_number(row["agent_wait_minutes"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Cause ledger",
            "",
            (
                "| Cause | Candidates | Runs | Attempts | CI/Hull job-min | "
                "Expansion job-min | All hosted job-min | Workflow wall min | "
                "Agent waits | Agent wait min |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["causes"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["cause"],
                    str(row["candidate_count"]),
                    str(row["workflow_run_count"]),
                    str(row["workflow_attempt_count"]),
                    _format_number(row["ci_hull_raw_job_minutes"]),
                    _format_number(
                        row["package_expansion_raw_job_minutes"]
                    ),
                    _format_number(row["all_hosted_raw_job_minutes"]),
                    _format_number(row["summed_workflow_wall_minutes"]),
                    str(row["agent_wait_count"]),
                    _format_number(row["agent_wait_minutes"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Candidate attribution",
            "",
            (
                "| PR | Ordinal | Head | Candidate | First observed | Cause | "
                "Basis | Runs | Raw job-min |"
            ),
            "| ---: | ---: | --- | --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for row in report["candidate_ledger"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"#{row['pr_number']}",
                    str(row["ordinal"]),
                    f"`{row['head_sha'][:12]}`",
                    f"`{row['candidate_sha'][:12]}`",
                    row["first_observed_at"],
                    row["cause"],
                    row["attribution"],
                    ", ".join(str(value) for value in row["workflow_run_ids"]),
                    _format_number(row["raw_job_minutes"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Workflow run attribution",
            "",
            (
                "| Run | PR | PR basis | Head | Workflow | Started | Cause | "
                "Cause basis | Attempts | Raw job-min | Wall min |"
            ),
            (
                "| ---: | ---: | --- | --- | --- | --- | --- | --- | ---: | "
                "---: | ---: |"
            ),
        ]
    )
    for row in report["run_ledger"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run_id"]),
                    f"#{row['pr_number']}",
                    row["pr_attribution"],
                    f"`{row['head_sha'][:12]}`",
                    row["workflow"],
                    row["started_at"],
                    row["cause"],
                    row["attribution"],
                    str(row["attempt_count"]),
                    _format_number(row["raw_job_minutes"]),
                    _format_number(row["workflow_wall_minutes"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Agent wait intervals",
            "",
        ]
    )
    if not report["agent_wait_ledger"]:
        lines.append(
            "No trace-derived intervals were supplied. The report does not "
            "substitute workflow duration for agent waiting."
        )
    else:
        lines.extend(
            [
                (
                    "| Wait ID | PR | Head | Start | End | Cause | Run IDs | "
                    "Wait min | Evidence |"
                ),
                "| --- | ---: | --- | --- | --- | --- | --- | ---: | --- |",
            ]
        )
        for row in report["agent_wait_ledger"]:
            head = (
                f"`{row['head_sha'][:12]}`"
                if row["head_sha"] is not None
                else "n/a"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["wait_id"],
                        f"#{row['pr_number']}",
                        head,
                        row["started_at"],
                        row["ended_at"],
                        row["cause"],
                        ", ".join(str(value) for value in row["run_ids"])
                        or "n/a",
                        _format_number(row["wait_minutes"]),
                        row["evidence"].replace("|", "\\|"),
                    ]
                )
                + " |"
            )

    if report["warnings"]:
        lines.extend(["", "## Evidence warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def _gh_api(endpoint: str) -> object:
    completed = subprocess.run(
        ["gh", "api", endpoint],
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise ReportError(f"gh api failed for {endpoint}: {message or 'no output'}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ReportError(
            f"gh api returned unreadable JSON for {endpoint}: {error}"
        ) from error


def _workflow_runs(
    api: Api,
    *,
    repository: str,
    workflow: str,
    since: datetime,
) -> list[Mapping[str, Any]]:
    collected: list[Mapping[str, Any]] = []
    for page in range(1, 101):
        payload = _mapping(
            api(
                f"repos/{repository}/actions/workflows/{workflow}/runs"
                f"?per_page=100&page={page}"
            ),
            f"{workflow} workflow response",
        )
        rows = [
            _mapping(row, f"{workflow} workflow run")
            for row in _sequence(
                payload.get("workflow_runs"), f"{workflow} workflow_runs"
            )
        ]
        if not rows:
            return collected
        for row in rows:
            created = _timestamp(
                row.get("created_at"), f"{workflow} run created_at"
            )
            if created >= since:
                collected.append(row)
        oldest = min(
            _timestamp(row.get("created_at"), "workflow run created_at")
            for row in rows
        )
        if oldest < since or len(rows) < 100:
            return collected
    raise ReportError(f"{workflow} workflow listing exceeded 100 pages")


def _jobs(api: Api, *, repository: str, run_id: int) -> list[Mapping[str, Any]]:
    collected: list[Mapping[str, Any]] = []
    for page in range(1, 101):
        payload = _mapping(
            api(
                f"repos/{repository}/actions/runs/{run_id}/jobs"
                f"?filter=all&per_page=100&page={page}"
            ),
            f"run {run_id} jobs response",
        )
        rows = [
            _mapping(row, f"run {run_id} job")
            for row in _sequence(payload.get("jobs"), f"run {run_id} jobs")
        ]
        collected.extend(rows)
        if len(rows) < 100:
            return collected
    raise ReportError(f"run {run_id} job listing exceeded 100 pages")


def _pull_number(run: Mapping[str, Any]) -> int | None:
    pulls = run.get("pull_requests")
    if not isinstance(pulls, Sequence):
        return None
    numbers = {
        pull.get("number")
        for pull in pulls
        if isinstance(pull, Mapping)
        and type(pull.get("number")) is int
        and pull["number"] > 0
    }
    if len(numbers) == 1:
        return numbers.pop()
    return None


def _pull_head(run: Mapping[str, Any]) -> str | None:
    pulls = run.get("pull_requests")
    if not isinstance(pulls, Sequence) or len(pulls) != 1:
        return None
    pull = pulls[0]
    if not isinstance(pull, Mapping):
        return None
    head = pull.get("head")
    if not isinstance(head, Mapping):
        return None
    sha = head.get("sha")
    return sha if isinstance(sha, str) and sha else None


def _run_head_key(run: Mapping[str, Any]) -> tuple[str, str] | None:
    branch = run.get("head_branch")
    repository = run.get("head_repository")
    if (
        not isinstance(branch, str)
        or not branch
        or not isinstance(repository, Mapping)
    ):
        return None
    full_name = repository.get("full_name")
    if not isinstance(full_name, str) or not full_name:
        return None
    return full_name, branch


def _branch_pull_number(
    run: Mapping[str, Any],
    pull_windows: Mapping[
        tuple[str, str], Sequence[tuple[int, datetime, datetime]]
    ],
) -> int | None:
    if run.get("event") != "pull_request":
        return None
    key = _run_head_key(run)
    if key is None:
        return None
    created_at = _timestamp(run.get("created_at"), "workflow run created_at")
    matches = [
        number
        for number, opened_at, updated_at in pull_windows.get(key, [])
        if opened_at <= created_at <= updated_at
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def collect_github_data(
    *,
    repository: str,
    prs: Sequence[int],
    attribution_payload: Mapping[str, Any] | None,
    since: datetime,
    api: Api = _gh_api,
) -> dict[str, Any]:
    cohort = set(prs)
    ledger = validate_ledger(
        attribution_payload if attribution_payload is not None else _empty_ledger()
    )
    run_attributions = {
        row["run_id"]: row for row in ledger["run_attributions"]
    }
    pulls: list[dict[str, Any]] = []
    pull_windows: dict[
        tuple[str, str], list[tuple[int, datetime, datetime]]
    ] = defaultdict(list)
    for pr_number in sorted(cohort):
        raw_pull = _mapping(
            api(f"repos/{repository}/pulls/{pr_number}"),
            f"pull request #{pr_number}",
        )
        if _positive_int(raw_pull.get("number"), "pull request number") != pr_number:
            raise ReportError(f"pull request endpoint did not return #{pr_number}")
        head = _mapping(raw_pull.get("head"), f"pull request #{pr_number} head")
        base = _mapping(raw_pull.get("base"), f"pull request #{pr_number} base")
        head_repository = _mapping(
            head.get("repo"), f"pull request #{pr_number} head repository"
        )
        head_repository_name = _text(
            head_repository.get("full_name"),
            f"pull request #{pr_number} head repository name",
        )
        head_ref = _text(head.get("ref"), f"pull request #{pr_number} head ref")
        opened_at = _timestamp(
            raw_pull.get("created_at"), f"pull request #{pr_number} created_at"
        )
        updated_at = _timestamp(
            raw_pull.get("updated_at"), f"pull request #{pr_number} updated_at"
        )
        pull_windows[(head_repository_name, head_ref)].append(
            (pr_number, opened_at, updated_at)
        )
        pulls.append(
            {
                "number": pr_number,
                "title": raw_pull.get("title"),
                "state": raw_pull.get("state"),
                "created_at": raw_pull.get("created_at"),
                "updated_at": raw_pull.get("updated_at"),
                "head_sha": head.get("sha"),
                "head_ref": head_ref,
                "head_repository": head_repository_name,
                "base_ref": base.get("ref"),
            }
        )
    all_runs: dict[int, Mapping[str, Any]] = {}
    run_pr_numbers: dict[int, int] = {}
    run_pr_bases: dict[int, str] = {}
    for workflow in (*CI_WORKFLOWS, EXPANSION_WORKFLOW):
        for run in _workflow_runs(
            api, repository=repository, workflow=Path(workflow).name, since=since
        ):
            run_id = _positive_int(run.get("id"), "workflow run id")
            pr_number = _pull_number(run)
            pr_basis = "workflow pull_requests"
            if pr_number is None:
                pr_number = _branch_pull_number(run, pull_windows)
                pr_basis = "exact head repository/ref and PR activity window"
            attribution = run_attributions.get(run_id)
            if (
                pr_number is not None
                and attribution is not None
                and pr_number != attribution["pr_number"]
            ):
                raise ReportError(
                    f"run {run_id} ledger PR disagrees with GitHub identity"
                )
            if pr_number not in cohort and (
                attribution is None or attribution["pr_number"] not in cohort
            ):
                continue
            if pr_number in cohort:
                run_pr_numbers[run_id] = pr_number
                run_pr_bases[run_id] = pr_basis
            elif attribution is not None:
                run_pr_numbers[run_id] = attribution["pr_number"]
                run_pr_bases[run_id] = "attribution ledger"
            all_runs[run_id] = run

    for run_id, attribution in run_attributions.items():
        if attribution["pr_number"] not in cohort or run_id in all_runs:
            continue
        all_runs[run_id] = _mapping(
            api(f"repos/{repository}/actions/runs/{run_id}"),
            f"attributed workflow run {run_id}",
        )
        run_pr_numbers[run_id] = attribution["pr_number"]
        run_pr_bases[run_id] = "attribution ledger"

    normalized: list[dict[str, Any]] = []
    for run_id, raw in sorted(
        all_runs.items(),
        key=lambda item: _timestamp(
            item[1].get("created_at"), f"run {item[0]} created_at"
        ),
    ):
        path = _workflow_path(raw)
        if path not in CI_WORKFLOWS and path != EXPANSION_WORKFLOW:
            raise ReportError(
                f"attributed run {run_id} uses unsupported workflow {path}"
            )
        attribution = run_attributions.get(run_id)
        pr_number = run_pr_numbers.get(run_id)
        if pr_number not in cohort:
            continue
        head_sha = _pull_head(raw)
        if (
            head_sha is None
            and raw.get("event") == "pull_request"
            and isinstance(raw.get("head_sha"), str)
        ):
            head_sha = raw["head_sha"]
        if head_sha is None and attribution is not None:
            head_sha = attribution["head_sha"]
        if (
            head_sha is not None
            and attribution is not None
            and raw.get("event") == "pull_request"
            and head_sha != attribution["head_sha"]
        ):
            raise ReportError(
                f"run {run_id} ledger head disagrees with GitHub identity"
            )
        if head_sha is None:
            raise ReportError(
                f"run {run_id} has no safely attributable PR head; add a "
                "run_attributions ledger row"
            )
        candidate_sha = raw.get("head_sha")
        if attribution is not None and attribution["candidate_sha"] is not None:
            candidate_sha = attribution["candidate_sha"]
        normalized.append(
            {
                "id": run_id,
                "run_attempt": raw.get("run_attempt", 1),
                "workflow_path": path,
                "workflow_name": raw.get("name"),
                "event": raw.get("event"),
                "pr_number": pr_number,
                "pr_attribution": run_pr_bases[run_id],
                "head_sha": head_sha,
                "candidate_sha": candidate_sha,
                "created_at": raw.get("created_at"),
                "run_started_at": raw.get("run_started_at"),
                "updated_at": raw.get("updated_at"),
                "status": raw.get("status"),
                "conclusion": raw.get("conclusion"),
                "jobs": _jobs(api, repository=repository, run_id=run_id),
            }
        )
    return {
        "schema": GITHUB_SCHEMA,
        "repository": repository,
        "collected_at": _timestamp_text(datetime.now(timezone.utc)),
        "collected_since": _timestamp_text(since),
        "pull_requests": pulls,
        "workflow_runs": normalized,
    }


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text()), label)
    except OSError as error:
        raise ReportError(f"cannot read {label} {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ReportError(f"{label} {path} is not valid JSON: {error}") from error


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prs",
        required=True,
        help=(
            "comma-separated PR numbers and inclusive ranges, for example "
            "2105,2111-2115"
        ),
    )
    parser.add_argument("--repository", help="OWNER/REPO for live GitHub queries")
    parser.add_argument(
        "--github-input",
        type=Path,
        help="normalized GitHub snapshot; omit to query with gh api",
    )
    parser.add_argument(
        "--attribution-ledger",
        type=Path,
        help="optional semantic-cause and agent-wait evidence",
    )
    parser.add_argument(
        "--since",
        help="earliest workflow creation timestamp for live collection",
    )
    parser.add_argument(
        "--snapshot-output",
        type=Path,
        help="write the normalized live GitHub response for repeatable reruns",
    )
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    try:
        prs = parse_prs(arguments.prs)
        attribution = (
            _read_json(arguments.attribution_ledger, "attribution ledger")
            if arguments.attribution_ledger
            else None
        )
        if arguments.github_input is not None:
            github_payload = _read_json(arguments.github_input, "GitHub input")
            repository = _text(
                github_payload.get("repository"), "GitHub input repository"
            )
            if (
                arguments.repository is not None
                and arguments.repository != repository
            ):
                raise ReportError(
                    "--repository does not match the GitHub input repository"
                )
        else:
            if arguments.repository is None:
                raise ReportError(
                    "--repository is required when --github-input is omitted"
                )
            if arguments.since is None:
                raise ReportError(
                    "--since is required for bounded live GitHub collection"
                )
            since = _timestamp(arguments.since, "--since")
            github_payload = collect_github_data(
                repository=arguments.repository,
                prs=prs,
                attribution_payload=attribution,
                since=since,
            )
            if arguments.snapshot_output is not None:
                _write_text(
                    arguments.snapshot_output,
                    json.dumps(github_payload, indent=2, sort_keys=True) + "\n",
                )
        report = build_report(
            github_payload,
            prs=prs,
            attribution_payload=attribution,
        )
        _write_text(
            arguments.json_output,
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        _write_text(arguments.markdown_output, render_markdown(report))
    except (ReportError, OSError, ValueError) as error:
        print(f"ci-pr-lifecycle-report: {error}", file=sys.stderr)
        return 2

    summary = report["summary"]
    print(
        "ci-pr-lifecycle-report: "
        f"{summary['candidate_count']} candidates, "
        f"{summary['package_expansion_count']} expansions, "
        f"{summary['cumulative_ci_hull_raw_job_minutes']} cumulative "
        "CI/Hull raw job-minutes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
