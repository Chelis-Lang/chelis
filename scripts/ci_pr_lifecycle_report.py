#!/usr/bin/env python3
"""Report full GitHub Actions and candidate-lifecycle work for a PR cohort.

The report separates per-job-rounded list-price estimates, raw execution,
workflow wall time, and agent waiting. GitHub supplies workflow/job evidence;
agent waiting and semantic causes that cannot be derived safely come from an
attribution ledger.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
from urllib.parse import urlencode
import zipfile


GITHUB_SCHEMA = "chelis-ci-pr-lifecycle-github-v1"
LEDGER_SCHEMA = "chelis-ci-pr-lifecycle-attribution-v1"
REPORT_SCHEMA = "chelis-ci-pr-lifecycle-report-v1"

CI_WORKFLOWS = {
    ".github/workflows/ci.yml": "CI",
    ".github/workflows/conformance.yml": "Hull",
}
EXPANSION_WORKFLOW = ".github/workflows/pr-package-expansion.yml"
EVENT_SIBLING_WINDOW_SECONDS = 120
WORKFLOW_RUN_PARENT_WINDOW_SECONDS = 15
WORKFLOW_RUN_PARENTS = {
    ".github/workflows/pr-candidate-receipt.yml": {
        "CI",
        "Hull Conformance",
        "Changelog",
        "PR Contract Acknowledgements",
        "PR Base Retarget Validation",
        "Secret scan",
    },
    ".github/workflows/openspec-autoland-controller.yml": {
        "OpenSpec autoland signal",
    },
}
DEFAULT_LINUX_X64_RATE_USD = 0.006
DEFAULT_LINUX_X64_SLIM_RATE_USD = 0.002
PRICING_AS_OF = "2026-09-16"
PRICING_SOURCE = (
    "https://docs.github.com/en/billing/reference/actions-minute-multipliers"
)
CAUSES = (
    "initial-candidate",
    "review-repair",
    "ordinary-content-push",
    "non-conflicting-rebase-base-update",
    "trivial-or-hand-resolved-conflict-rebase",
    "base-retarget-stack-collapse",
    "ci-policy-ci-repair",
    "pull-request-metadata-edit",
    "package-expansion-rerun",
    "unknown",
)


class ReportError(ValueError):
    """The requested report cannot be produced from the supplied evidence."""


Api = Callable[[str], object]
LogApi = Callable[[int], str | None]


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
    for key in ("created_at", "run_started_at", "updated_at"):
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
            or _is_non_vm_synthetic(job)
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
        if job.get("conclusion") == "skipped" or _is_non_vm_synthetic(job):
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
            or _is_non_vm_synthetic(job)
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
    jobs = _sequence(run.get("jobs", []), "workflow run jobs")
    for index, raw_job in enumerate(jobs):
        job = _mapping(raw_job, f"workflow run job {index}")
        if (
            job.get("conclusion") == "skipped"
            or _is_non_vm_synthetic(job)
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
    return windows


def _workflow_wall_seconds(run: Mapping[str, Any]) -> float:
    return sum(
        (completed - started).total_seconds()
        for started, completed in _attempt_windows(run).values()
    )


def _attempt_count(run: Mapping[str, Any]) -> int:
    attempts = {
        _positive_int(run.get("run_attempt", 1), "workflow run_attempt")
    }
    for index, raw_job in enumerate(
        _sequence(run.get("jobs", []), "workflow run jobs")
    ):
        job = _mapping(raw_job, f"workflow run job {index}")
        attempts.add(
            _positive_int(
                job.get("run_attempt", run.get("run_attempt", 1)),
                "workflow job run_attempt",
            )
        )
    return max(attempts)


def _job_labels(job: Mapping[str, Any]) -> set[str]:
    labels = job.get("labels", [])
    if not isinstance(labels, Sequence) or isinstance(
        labels, (str, bytes, bytearray)
    ):
        raise ReportError("workflow job labels must be an array")
    return {
        label.strip().lower()
        for label in labels
        if isinstance(label, str) and label.strip()
    }


def _is_non_vm_synthetic(job: Mapping[str, Any]) -> bool:
    return (
        job.get("started_at") is not None
        and not _job_labels(job)
        and job.get("runner_id") is None
        and job.get("runner_name") is None
        and not job.get("steps")
    )


def _runner_kind(job: Mapping[str, Any]) -> str:
    labels = _job_labels(job)
    if "self-hosted" in labels:
        return "self-hosted"
    if "ubuntu-slim" in labels:
        return "linux-x64-slim"
    ubuntu = [label for label in labels if label.startswith("ubuntu-")]
    if ubuntu and not any("arm" in label for label in ubuntu):
        return "linux-x64-standard"
    return "unpriced"


def _job_accounting(
    run: Mapping[str, Any],
    *,
    linux_x64_rate_usd: float,
    linux_x64_slim_rate_usd: float,
) -> dict[str, int | float]:
    if linux_x64_rate_usd < 0:
        raise ReportError("Linux x64 minute rate must not be negative")
    if linux_x64_slim_rate_usd < 0:
        raise ReportError("Linux x64 slim minute rate must not be negative")
    result: dict[str, int | float] = {
        "started_job_count": 0,
        "skipped_job_count": 0,
        "never_started_job_count": 0,
        "non_vm_synthetic_check_count": 0,
        "canceled_started_job_count": 0,
        "failed_started_job_count": 0,
        "linux_x64_started_job_count": 0,
        "linux_x64_standard_started_job_count": 0,
        "linux_x64_slim_started_job_count": 0,
        "self_hosted_started_job_count": 0,
        "unpriced_started_job_count": 0,
        "estimated_billable_linux_x64_minutes": 0,
        "estimated_billable_linux_x64_standard_minutes": 0,
        "estimated_billable_linux_x64_slim_minutes": 0,
        "estimated_list_price_usd": 0.0,
    }
    observed_attempts: set[int] = set()
    for index, raw_job in enumerate(
        _sequence(run.get("jobs", []), "workflow run jobs")
    ):
        job = _mapping(raw_job, f"workflow run job {index}")
        observed_attempts.add(
            _positive_int(
                job.get("run_attempt", run.get("run_attempt", 1)),
                "workflow job run_attempt",
            )
        )
        conclusion = str(job.get("conclusion") or "").lower()
        if conclusion == "skipped":
            result["skipped_job_count"] += 1
            continue
        if job.get("started_at") is None:
            result["never_started_job_count"] += 1
            continue
        if _is_non_vm_synthetic(job):
            result["non_vm_synthetic_check_count"] += 1
            continue

        result["started_job_count"] += 1
        if conclusion in {"cancelled", "canceled"}:
            result["canceled_started_job_count"] += 1
        if conclusion in {"failure", "timed_out", "startup_failure"}:
            result["failed_started_job_count"] += 1

        kind = _runner_kind(job)
        if kind == "self-hosted":
            result["self_hosted_started_job_count"] += 1
            continue
        if kind not in {"linux-x64-standard", "linux-x64-slim"}:
            result["unpriced_started_job_count"] += 1
            continue

        result["linux_x64_started_job_count"] += 1
        if kind == "linux-x64-slim":
            result["linux_x64_slim_started_job_count"] += 1
        else:
            result["linux_x64_standard_started_job_count"] += 1
        billed_minutes = 1
        if job.get("completed_at") is not None:
            started = _timestamp(job["started_at"], "workflow job started_at")
            completed = _timestamp(
                job["completed_at"], "workflow job completed_at"
            )
            if completed >= started:
                billed_minutes = max(
                    1,
                    math.ceil((completed - started).total_seconds() / 60.0),
                )
        result["estimated_billable_linux_x64_minutes"] += billed_minutes
        if kind == "linux-x64-slim":
            result["estimated_billable_linux_x64_slim_minutes"] += billed_minutes
        else:
            result[
                "estimated_billable_linux_x64_standard_minutes"
            ] += billed_minutes

    result["never_started_job_count"] += max(
        0, _attempt_count(run) - len(observed_attempts)
    )

    result["estimated_list_price_usd"] = round(
        int(result["estimated_billable_linux_x64_standard_minutes"])
        * linux_x64_rate_usd
        + int(result["estimated_billable_linux_x64_slim_minutes"])
        * linux_x64_slim_rate_usd,
        6,
    )
    return result


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
    candidate_keys: set[tuple[int, str, str]] = set()
    for index, raw in enumerate(
        _sequence(
            payload.get("candidate_attributions", []),
            "candidate_attributions",
        )
    ):
        row = _mapping(raw, f"candidate attribution {index}")
        pr_number = _positive_int(row.get("pr_number"), "candidate PR number")
        head_sha = _text(row.get("head_sha"), "candidate head_sha")
        candidate_sha = _text(
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
        head_sha = _optional_text(
            row.get("head_sha"), "agent wait head_sha"
        )
        candidate_sha = _optional_text(
            row.get("candidate_sha"), "agent wait candidate_sha"
        )
        if head_sha is None and candidate_sha is None:
            raise ReportError(
                f"agent wait {wait_id!r} requires head_sha or candidate_sha"
            )
        run_ids_value = [
            _positive_int(run_id, f"agent wait {wait_id} run id")
            for run_id in _sequence(
                row.get("run_ids", []), f"agent wait {wait_id} run_ids"
            )
        ]
        if len(run_ids_value) != len(set(run_ids_value)):
            raise ReportError(
                f"agent wait {wait_id!r} contains duplicate run IDs"
            )
        if not run_ids_value:
            raise ReportError(
                f"agent wait {wait_id!r} requires at least one workflow run ID"
            )
        agent_waits.append(
            {
                "wait_id": wait_id,
                "agent_id": _text(
                    row.get("agent_id"), "agent wait agent_id"
                ),
                "pr_number": _positive_int(
                    row.get("pr_number"), "agent wait PR number"
                ),
                "head_sha": head_sha,
                "candidate_sha": candidate_sha,
                "cause": _validate_cause(row.get("cause"), "agent wait cause"),
                "started_at": _timestamp_text(started_at),
                "ended_at": _timestamp_text(ended_at),
                "run_ids": run_ids_value,
                "evidence": _text(row.get("evidence"), "agent wait evidence"),
                "note": _optional_text(row.get("note"), "agent wait note"),
            }
        )

    waits_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in agent_waits:
        waits_by_agent[row["agent_id"]].append(row)
    for agent_id, rows in waits_by_agent.items():
        rows.sort(key=lambda row: _timestamp(row["started_at"], "agent wait start"))
        for previous, current in zip(rows, rows[1:]):
            previous_end = _timestamp(previous["ended_at"], "agent wait end")
            current_start = _timestamp(current["started_at"], "agent wait start")
            if current_start < previous_end:
                raise ReportError(
                    f"agent {agent_id!r} has overlapping waits "
                    f"{previous['wait_id']!r} and {current['wait_id']!r}"
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
    used_rows: set[int],
) -> dict[str, str | None]:
    matches = [
        (index, row)
        for index, row in enumerate(ledger["candidate_attributions"])
        if row["pr_number"] == pr_number
        and row["head_sha"] == head_sha
        and row["candidate_sha"] == candidate_sha
    ]
    if len(matches) > 1:
        raise ReportError(
            f"multiple ledger rows match PR #{pr_number} candidate {candidate_sha}"
        )
    if matches:
        index, row = matches[0]
        used_rows.add(index)
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
    source_rows: Sequence[Mapping[str, Any]],
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
    if source_rows:
        causes = {row["cause"] for row in source_rows}
        if len(causes) == 1:
            cause = causes.pop()
            return {
                "cause": cause,
                "attribution": "inferred-workflow-run-parent",
                "evidence": (
                    "workflow_run child of run IDs "
                    + ", ".join(str(row["run_id"]) for row in source_rows)
                ),
                "note": None,
            }
    if expansion_ordinal is not None:
        if expansion_ordinal > 1:
            return {
                "cause": "package-expansion-rerun",
                "attribution": "inferred",
                "evidence": "later package-expansion run observed for the same PR",
                "note": None,
            }
        return {
            "cause": "unknown",
            "attribution": "unknown",
            "evidence": "no supplied ledger row establishes this expansion's cause",
            "note": None,
        }
    if candidate is not None:
        return {
            "cause": candidate["cause"],
            "attribution": candidate["attribution"],
            "evidence": candidate["evidence"],
            "note": candidate["note"],
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
    supplied_candidate_sha = run.get("candidate_sha")
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
        and run.get("event") in {
            "pull_request",
            "pull_request_target",
            "push",
        }
        and isinstance(supplied_head_sha, str)
        and supplied_head_sha != attribution["head_sha"]
    ):
        raise ReportError(
            f"run {run_id} head attribution disagrees with supplied GitHub data"
        )
    if (
        attribution is not None
        and run.get("event")
        in {"pull_request", "pull_request_target", "push"}
        and isinstance(supplied_candidate_sha, str)
        and attribution["candidate_sha"] is not None
        and supplied_candidate_sha != attribution["candidate_sha"]
    ):
        raise ReportError(
            f"run {run_id} candidate attribution disagrees with supplied GitHub data"
        )
    pr_number = _run_pr_number(run, run_attributions)
    if pr_number is not None:
        run["pr_number"] = pr_number
    if attribution is not None:
        run["head_sha"] = attribution["head_sha"]
        if (
            attribution["candidate_sha"] is not None
            and not isinstance(supplied_candidate_sha, str)
        ):
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
            "closed_at": (
                _timestamp_text(
                    _timestamp(pull.get("closed_at"), "pull request closed_at")
                )
                if pull.get("closed_at") is not None
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
    linux_x64_rate_usd: float = DEFAULT_LINUX_X64_RATE_USD,
    linux_x64_slim_rate_usd: float = DEFAULT_LINUX_X64_SLIM_RATE_USD,
) -> dict[str, Any]:
    if github_payload.get("schema") != GITHUB_SCHEMA:
        raise ReportError(f"GitHub input schema must be {GITHUB_SCHEMA}")
    repository = _text(github_payload.get("repository"), "repository")
    cohort = sorted({_positive_int(value, "PR number") for value in prs})
    if not cohort:
        raise ReportError("PR cohort must not be empty")
    generated = (
        _timestamp(generated_at, "generated_at")
        if generated_at is not None
        else datetime.now(timezone.utc)
    )
    collected = (
        _timestamp(github_payload.get("collected_at"), "collected_at")
        if github_payload.get("collected_at") is not None
        else generated
    )
    evidence_cutoff = min(generated, collected)
    pull_requests = _pull_requests(github_payload, cohort)
    ledger = validate_ledger(
        attribution_payload if attribution_payload is not None else _empty_ledger()
    )
    run_attributions = {
        row["run_id"]: row for row in ledger["run_attributions"]
    }
    runs: list[dict[str, Any]] = []
    seen_run_ids: set[int] = set()
    for index, raw in enumerate(
        _sequence(github_payload.get("workflow_runs"), "workflow_runs")
    ):
        normalized = _normalized_run(
            _mapping(raw, f"workflow run {index}"), run_attributions
        )
        run_id = _positive_int(normalized.get("id"), "workflow run id")
        if run_id in seen_run_ids:
            raise ReportError(f"duplicate workflow run ID {run_id}")
        seen_run_ids.add(run_id)
        runs.append(normalized)
    missing_attributed_runs = sorted(
        row["run_id"]
        for row in ledger["run_attributions"]
        if row["pr_number"] in cohort and row["run_id"] not in seen_run_ids
    )
    if missing_attributed_runs:
        raise ReportError(
            "attribution ledger references workflow runs missing from the "
            f"snapshot: {missing_attributed_runs}"
        )
    runs = [
        run
        for run in runs
        if run.get("pr_number") in cohort
    ]
    collected_since = (
        _timestamp(github_payload.get("collected_since"), "collected_since")
        if github_payload.get("collected_since") is not None
        else None
    )
    for run in runs:
        run_id = _positive_int(run.get("id"), "workflow run id")
        attribution = run_attributions.get(run_id)
        if attribution is None:
            continue
        run_created = _run_time(run)
        if collected_since is not None and run_created < collected_since:
            raise ReportError(
                f"attributed workflow run {run_id} predates collected_since"
            )
        pull = pull_requests[attribution["pr_number"]]
        pull_created = _timestamp(
            pull["created_at"],
            f"pull request #{attribution['pr_number']} created_at",
        )
        pull_closed = (
            _timestamp(
                pull["closed_at"],
                f"pull request #{attribution['pr_number']} closed_at",
            )
            if pull["closed_at"] is not None
            else evidence_cutoff
        )
        if not pull_created <= run_created <= pull_closed:
            raise ReportError(
                f"attributed workflow run {run_id} falls outside PR "
                f"#{attribution['pr_number']} lifetime"
            )

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
    candidate_by_head: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    candidate_seconds: dict[tuple[int, str], tuple[float, float]] = {}
    used_candidate_rows: set[int] = set()
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
                used_rows=used_candidate_rows,
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
            candidate_by_head[(pr_number, head_sha)].append(row)
            candidate_seconds[key] = (job_seconds, wall_seconds)

    unused_candidate_rows = [
        row
        for index, row in enumerate(ledger["candidate_attributions"])
        if row["pr_number"] in cohort and index not in used_candidate_rows
    ]
    if unused_candidate_rows:
        identities = [
            (
                row["pr_number"],
                row["head_sha"],
                row["candidate_sha"],
            )
            for row in unused_candidate_rows
        ]
        raise ReportError(
            f"unused candidate attribution rows for cohort: {identities}"
        )

    expansion_by_pr: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for run in expansion_runs:
        pr_number = _positive_int(run.get("pr_number"), "expansion run PR number")
        expansion_by_pr[pr_number].append(run)
    for grouped in expansion_by_pr.values():
        grouped.sort(key=_run_time)

    run_rows: list[dict[str, Any]] = []
    built_run_rows_by_id: dict[int, dict[str, Any]] = {}
    run_seconds: dict[int, tuple[float, float]] = {}
    run_accounting: dict[int, dict[str, int | float]] = {}
    for run in sorted(runs, key=_run_time):
        run_id = _positive_int(run.get("id"), "workflow run id")
        pr_number = _positive_int(run.get("pr_number"), "workflow run PR number")
        path = _workflow_path(run)
        if path in CI_WORKFLOWS:
            candidate = candidate_by_key.get((pr_number, _candidate_id(run)))
        else:
            run_created = _run_time(run)
            head_candidates = [
                row
                for row in candidate_by_head.get(
                    (pr_number, _head_sha(run)), []
                )
                if abs(
                    (
                        run_created
                        - _timestamp(
                            row["first_observed_at"],
                            "candidate first_observed_at",
                        )
                    ).total_seconds()
                )
                <= EVENT_SIBLING_WINDOW_SECONDS
            ]
            candidate = head_candidates[0] if len(head_candidates) == 1 else None
        expansion_ordinal = None
        if path == EXPANSION_WORKFLOW:
            expansion_ordinal = (
                expansion_by_pr[pr_number].index(run) + 1
            )
        source_run_ids = [
            _positive_int(value, "workflow source run id")
            for value in _sequence(
                run.get("source_run_ids", []), "workflow source_run_ids"
            )
        ]
        if len(source_run_ids) != len(set(source_run_ids)):
            raise ReportError(
                f"run {run_id} contains duplicate source workflow run IDs"
            )
        source_rows: list[Mapping[str, Any]] = []
        for source_run_id in source_run_ids:
            source = built_run_rows_by_id.get(source_run_id)
            if source is None:
                raise ReportError(
                    f"run {run_id} references unavailable source workflow "
                    f"run {source_run_id}"
                )
            if source["pr_number"] != pr_number:
                raise ReportError(
                    f"run {run_id} source run {source_run_id} belongs to a "
                    "different pull request"
                )
            if source["head_sha"] != _head_sha(run):
                raise ReportError(
                    f"run {run_id} source run {source_run_id} has a different "
                    "pull-request head"
                )
            source_rows.append(source)
        attribution = _run_attribution(
            run,
            candidate=candidate,
            expansion_ordinal=expansion_ordinal,
            ledger=ledger,
            source_rows=source_rows,
        )
        pr_attribution = _optional_text(
            run.get("pr_attribution"), "workflow run PR attribution"
        )
        job_seconds = _job_seconds(run)
        wall_seconds = _workflow_wall_seconds(run)
        accounting = _job_accounting(
            run,
            linux_x64_rate_usd=linux_x64_rate_usd,
            linux_x64_slim_rate_usd=linux_x64_slim_rate_usd,
        )
        run_seconds[run_id] = (job_seconds, wall_seconds)
        run_accounting[run_id] = accounting
        workflow_name = _optional_text(
            run.get("workflow_name"), "workflow run name"
        )
        run_row = {
                "run_id": run_id,
                "pr_number": pr_number,
                "pr_attribution": (
                    pr_attribution
                    if pr_attribution is not None
                    else "provided GitHub input"
                ),
                "head_sha": _head_sha(run),
                "candidate_sha": (
                    _candidate_id(run)
                    if path in CI_WORKFLOWS
                    else (
                        candidate["candidate_sha"]
                        if candidate is not None
                        else None
                    )
                ),
                "workflow": (
                    CI_WORKFLOWS[path]
                    if path in CI_WORKFLOWS
                    else (
                        "Package Expansion"
                        if path == EXPANSION_WORKFLOW
                        else (workflow_name or Path(path).stem)
                    )
                ),
                "workflow_path": path,
                "event": _optional_text(run.get("event"), "workflow run event"),
                "created_at": _timestamp_text(_run_time(run)),
                "started_at": (
                    _timestamp_text(
                        _timestamp(
                            run["run_started_at"],
                            "workflow run run_started_at",
                        )
                    )
                    if run.get("run_started_at") is not None
                    else None
                ),
                "completed_at": (
                    _timestamp_text(
                        _timestamp(
                            run["updated_at"],
                            "workflow run updated_at",
                        )
                    )
                    if run.get("updated_at") is not None
                    else None
                ),
                "cause": attribution["cause"],
                "attribution": attribution["attribution"],
                "evidence": attribution["evidence"],
                "note": attribution["note"],
                "attempt_count": _attempt_count(run),
                "raw_job_minutes": _minutes(job_seconds),
                "workflow_wall_minutes": _minutes(wall_seconds),
                "untimed_job_count": _untimed_job_count(run),
                "timing_anomaly_count": _timing_anomaly_count(run),
                **accounting,
            }
        run_rows.append(run_row)
        built_run_rows_by_id[run_id] = run_row

    wait_rows: list[dict[str, Any]] = []
    wait_seconds_by_id: dict[str, float] = {}
    run_rows_by_id = {row["run_id"]: row for row in run_rows}
    for row in ledger["agent_waits"]:
        if row["pr_number"] not in cohort:
            continue
        wait_started = _timestamp(
            row["started_at"], f"agent wait {row['wait_id']} started_at"
        )
        wait_ended = _timestamp(
            row["ended_at"], f"agent wait {row['wait_id']} ended_at"
        )
        if wait_ended > evidence_cutoff:
            raise ReportError(
                f"agent wait {row['wait_id']!r} ends after the report evidence "
                "cutoff"
            )
        pull_created = _timestamp(
            pull_requests[row["pr_number"]]["created_at"],
            f"pull request #{row['pr_number']} created_at",
        )
        if wait_started < pull_created:
            raise ReportError(
                f"agent wait {row['wait_id']!r} starts before PR "
                f"#{row['pr_number']} was created"
            )
        linked_windows: list[tuple[datetime, datetime]] = []
        linked_causes: set[str] = set()
        for run_id in row["run_ids"]:
            linked_run = run_rows_by_id.get(run_id)
            if linked_run is None:
                raise ReportError(
                    f"agent wait {row['wait_id']!r} references unknown "
                    f"workflow run {run_id}"
                )
            if linked_run["pr_number"] != row["pr_number"]:
                raise ReportError(
                    f"agent wait {row['wait_id']!r} run {run_id} belongs to "
                    f"PR #{linked_run['pr_number']}, not PR #{row['pr_number']}"
                )
            if (
                row["head_sha"] is not None
                and linked_run["head_sha"] != row["head_sha"]
            ):
                raise ReportError(
                    f"agent wait {row['wait_id']!r} run {run_id} does not "
                    "match its head_sha"
                )
            run_candidate = (
                linked_run["candidate_sha"] or linked_run["head_sha"]
            )
            if (
                row["candidate_sha"] is not None
                and run_candidate != row["candidate_sha"]
            ):
                raise ReportError(
                    f"agent wait {row['wait_id']!r} run {run_id} does not "
                    "match its candidate_sha"
                )
            if linked_run["cause"] != "unknown":
                linked_causes.add(linked_run["cause"])
            run_started = _timestamp(
                linked_run["created_at"],
                f"workflow run {run_id} created_at",
            )
            run_ended = (
                _timestamp(
                    linked_run["completed_at"],
                    f"workflow run {run_id} completed_at",
                )
                if linked_run["completed_at"] is not None
                else evidence_cutoff
            )
            linked_windows.append((run_started, run_ended))
        if not any(
            wait_started <= run_ended and wait_ended >= run_started
            for run_started, run_ended in linked_windows
        ):
            raise ReportError(
                f"agent wait {row['wait_id']!r} does not overlap any linked "
                "workflow run"
            )
        if linked_causes and row["cause"] not in linked_causes:
            raise ReportError(
                f"agent wait {row['wait_id']!r} cause disagrees with every "
                "linked workflow run"
            )
        wait_seconds = _duration_seconds(
            row["started_at"],
            row["ended_at"],
            f"agent wait {row['wait_id']}",
        )
        wait_seconds_by_id[row["wait_id"]] = wait_seconds
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
        ci_seconds = sum(run_seconds[row["run_id"]][0] for row in ci_runs)
        expansion_seconds = sum(
            run_seconds[row["run_id"]][0] for row in expansion_cause_runs
        )
        hosted_seconds = sum(
            run_seconds[row["run_id"]][0] for row in cause_runs
        )
        workflow_wall_seconds = sum(
            run_seconds[row["run_id"]][1] for row in cause_runs
        )
        agent_wait_seconds = sum(
            wait_seconds_by_id[row["wait_id"]] for row in cause_waits
        )
        estimated_billable_minutes = sum(
            int(
                run_accounting[row["run_id"]][
                    "estimated_billable_linux_x64_minutes"
                ]
            )
            for row in cause_runs
        )
        cause_rows.append(
            {
                "cause": cause,
                "candidate_count": len(cause_candidates),
                "workflow_run_count": len(cause_runs),
                "workflow_attempt_count": sum(
                    row["attempt_count"] for row in cause_runs
                ),
                "ci_hull_raw_job_minutes": _minutes(ci_seconds),
                "package_expansion_raw_job_minutes": _minutes(
                    expansion_seconds
                ),
                "all_hosted_raw_job_minutes": _minutes(hosted_seconds),
                "summed_workflow_wall_minutes": _minutes(
                    workflow_wall_seconds
                ),
                "agent_wait_count": len(cause_waits),
                "agent_wait_minutes": _minutes(agent_wait_seconds),
                "started_job_count": sum(
                    int(run_accounting[row["run_id"]]["started_job_count"])
                    for row in cause_runs
                ),
                "skipped_job_count": sum(
                    int(run_accounting[row["run_id"]]["skipped_job_count"])
                    for row in cause_runs
                ),
                "never_started_job_count": sum(
                    int(
                        run_accounting[row["run_id"]][
                            "never_started_job_count"
                        ]
                    )
                    for row in cause_runs
                ),
                "non_vm_synthetic_check_count": sum(
                    int(
                        run_accounting[row["run_id"]][
                            "non_vm_synthetic_check_count"
                        ]
                    )
                    for row in cause_runs
                ),
                "canceled_started_job_count": sum(
                    int(
                        run_accounting[row["run_id"]][
                            "canceled_started_job_count"
                        ]
                    )
                    for row in cause_runs
                ),
                "failed_started_job_count": sum(
                    int(
                        run_accounting[row["run_id"]][
                            "failed_started_job_count"
                        ]
                    )
                    for row in cause_runs
                ),
                "estimated_billable_linux_x64_minutes": estimated_billable_minutes,
                "estimated_list_price_usd": round(
                    sum(
                        float(
                            run_accounting[row["run_id"]][
                                "estimated_list_price_usd"
                            ]
                        )
                        for row in cause_runs
                    ),
                    6,
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
        cumulative_seconds = sum(
            candidate_seconds[(pr_number, row["candidate_sha"])][0]
            for row in pr_candidates
        )
        latest_seconds = (
            candidate_seconds[(pr_number, latest["candidate_sha"])][0]
            if latest
            else 0.0
        )
        pr_expansions = [
            row
            for row in run_rows
            if row["pr_number"] == pr_number
            and row["workflow_path"] == EXPANSION_WORKFLOW
        ]
        pr_waits = [row for row in wait_rows if row["pr_number"] == pr_number]
        pr_runs = [
            row for row in run_rows if row["pr_number"] == pr_number
        ]
        expansion_seconds = sum(
            run_seconds[row["run_id"]][0] for row in pr_expansions
        )
        agent_wait_seconds = sum(
            wait_seconds_by_id[row["wait_id"]] for row in pr_waits
        )
        estimated_billable_minutes = sum(
            int(
                run_accounting[row["run_id"]][
                    "estimated_billable_linux_x64_minutes"
                ]
            )
            for row in pr_runs
        )
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
                "cumulative_ci_hull_raw_job_minutes": _minutes(
                    cumulative_seconds
                ),
                "latest_candidate_ci_hull_raw_job_minutes": _minutes(
                    latest_seconds
                ),
                "amplification_ratio": _ratio(
                    cumulative_seconds, latest_seconds
                ),
                "package_expansion_raw_job_minutes": _minutes(
                    expansion_seconds
                ),
                "agent_wait_minutes": _minutes(agent_wait_seconds),
                "actions_workflow_run_count": len(pr_runs),
                "started_job_count": sum(
                    int(run_accounting[row["run_id"]]["started_job_count"])
                    for row in pr_runs
                ),
                "failed_started_job_count": sum(
                    int(
                        run_accounting[row["run_id"]][
                            "failed_started_job_count"
                        ]
                    )
                    for row in pr_runs
                ),
                "canceled_started_job_count": sum(
                    int(
                        run_accounting[row["run_id"]][
                            "canceled_started_job_count"
                        ]
                    )
                    for row in pr_runs
                ),
                "skipped_job_count": sum(
                    int(run_accounting[row["run_id"]]["skipped_job_count"])
                    for row in pr_runs
                ),
                "never_started_job_count": sum(
                    int(
                        run_accounting[row["run_id"]][
                            "never_started_job_count"
                        ]
                    )
                    for row in pr_runs
                ),
                "non_vm_synthetic_check_count": sum(
                    int(
                        run_accounting[row["run_id"]][
                            "non_vm_synthetic_check_count"
                        ]
                    )
                    for row in pr_runs
                ),
                "estimated_billable_linux_x64_minutes": estimated_billable_minutes,
                "estimated_list_price_usd": round(
                    sum(
                        float(
                            run_accounting[row["run_id"]][
                                "estimated_list_price_usd"
                            ]
                        )
                        for row in pr_runs
                    ),
                    6,
                ),
                "latest_head_sha": latest["head_sha"] if latest else None,
                "latest_candidate_sha": (
                    latest["candidate_sha"] if latest else None
                ),
            }
        )

    cumulative_seconds = sum(
        seconds[0] for seconds in candidate_seconds.values()
    )
    latest_seconds = sum(
        candidate_seconds[(row["pr_number"], row["latest_candidate_sha"])][0]
        for row in pr_rows
        if row["latest_candidate_sha"] is not None
    )
    implementation_run_rows = [
        row for row in run_rows if row["workflow_path"] in CI_WORKFLOWS
    ]
    expansion_run_rows = [
        row for row in run_rows if row["workflow_path"] == EXPANSION_WORKFLOW
    ]
    package_expansion_seconds = sum(
        run_seconds[row["run_id"]][0] for row in expansion_run_rows
    )
    all_hosted_seconds = sum(seconds[0] for seconds in run_seconds.values())
    all_workflow_wall_seconds = sum(
        seconds[1] for seconds in run_seconds.values()
    )
    all_agent_wait_seconds = sum(wait_seconds_by_id.values())
    all_estimated_billable_minutes = sum(
        int(
            accounting["estimated_billable_linux_x64_minutes"]
        )
        for accounting in run_accounting.values()
    )
    workflow_rows: list[dict[str, Any]] = []
    workflow_paths = sorted({row["workflow_path"] for row in run_rows})
    for workflow_path in workflow_paths:
        workflow_runs = [
            row for row in run_rows if row["workflow_path"] == workflow_path
        ]
        billable_minutes = sum(
            int(
                run_accounting[row["run_id"]][
                    "estimated_billable_linux_x64_minutes"
                ]
            )
            for row in workflow_runs
        )
        workflow_rows.append(
            {
                "workflow": workflow_runs[0]["workflow"],
                "workflow_path": workflow_path,
                "workflow_run_count": len(workflow_runs),
                "workflow_attempt_count": sum(
                    row["attempt_count"] for row in workflow_runs
                ),
                "started_job_count": sum(
                    int(run_accounting[row["run_id"]]["started_job_count"])
                    for row in workflow_runs
                ),
                "failed_started_job_count": sum(
                    int(
                        run_accounting[row["run_id"]][
                            "failed_started_job_count"
                        ]
                    )
                    for row in workflow_runs
                ),
                "canceled_started_job_count": sum(
                    int(
                        run_accounting[row["run_id"]][
                            "canceled_started_job_count"
                        ]
                    )
                    for row in workflow_runs
                ),
                "skipped_job_count": sum(
                    int(run_accounting[row["run_id"]]["skipped_job_count"])
                    for row in workflow_runs
                ),
                "never_started_job_count": sum(
                    int(
                        run_accounting[row["run_id"]][
                            "never_started_job_count"
                        ]
                    )
                    for row in workflow_runs
                ),
                "non_vm_synthetic_check_count": sum(
                    int(
                        run_accounting[row["run_id"]][
                            "non_vm_synthetic_check_count"
                        ]
                    )
                    for row in workflow_runs
                ),
                "raw_job_minutes": _minutes(
                    sum(
                        run_seconds[row["run_id"]][0]
                        for row in workflow_runs
                    )
                ),
                "estimated_billable_linux_x64_minutes": billable_minutes,
                "estimated_list_price_usd": round(
                    sum(
                        float(
                            run_accounting[row["run_id"]][
                                "estimated_list_price_usd"
                            ]
                        )
                        for row in workflow_runs
                    ),
                    6,
                ),
            }
        )
    warnings: list[str] = []
    if not ledger["manual_run_scope_complete"]:
        warnings.append(
            "Association-less workflow attribution is not declared complete; "
            "manual dispatch and other unassociated run counts are lower bounds "
            "until every cohort run has a run_attributions row."
        )
    collected_since_value = github_payload.get("collected_since")
    if collected_since_value is not None:
        since = _timestamp(collected_since_value, "collected_since")
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
    if any(row["attribution"] == "unknown" for row in run_rows):
        warnings.append(
            "Some workflow-run causes are unknown; supply run_attributions or "
            "candidate_attributions for semantic classification."
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
            "were excluded from raw job-minute totals; started recognized Linux "
            "jobs contribute a one-minute minimum to the billing estimate."
        )
    timing_anomalies = sum(row["timing_anomaly_count"] for row in run_rows)
    if timing_anomalies:
        warnings.append(
            f"{timing_anomalies} non-skipped jobs had inverted timestamps and "
            "were excluded from raw job-minute and workflow-wall totals; "
            "started recognized Linux jobs contribute a one-minute minimum to "
            "the billing estimate."
        )
    unpriced_started_jobs = sum(
        int(accounting["unpriced_started_job_count"])
        for accounting in run_accounting.values()
    )
    if unpriced_started_jobs:
        warnings.append(
            f"{unpriced_started_jobs} started jobs did not have a recognized "
            "standard or slim Linux x64 label and are not included in the list-price "
            "estimate."
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
        "pricing": {
            "currency": "USD",
            "pricing_as_of": PRICING_AS_OF,
            "source": PRICING_SOURCE,
            "linux_x64_minute_rate_usd": linux_x64_rate_usd,
            "linux_x64_slim_minute_rate_usd": linux_x64_slim_rate_usd,
            "method": (
                "Each started standard or slim Linux x64 job is rounded up "
                "separately to a whole minute and priced at its runner SKU. "
                "Skipped and never-started jobs cost zero. Started jobs with "
                "missing or inverted completion timestamps are estimated at "
                "one minute."
            ),
        },
        "definitions": {
            "candidate_count": (
                "Distinct implementation candidate identities observed across "
                "CI and Hull runs; candidate_sha is preferred over head_sha."
            ),
            "raw_job_minutes": (
                "Sum of completed_at-started_at for started jobs with valid "
                "timestamps. This is separate from billing-rounded time."
            ),
            "workflow_wall_minutes": (
                "Sum of each workflow attempt's earliest-job to latest-job "
                "window. Concurrent attempts and workflows can overlap."
            ),
                "agent_wait_minutes": (
                    "Sum of explicit attribution-ledger intervals during which an "
                    "agent was waiting for CI. It is never inferred from GitHub, "
                    "and supplied run IDs are checked against the PR and head or "
                    "synthetic candidate identity."
                ),
            "amplification_ratio": (
                "Cumulative CI/Hull raw job-minutes divided by the sum of each "
                "PR's latest-candidate CI/Hull raw job-minutes."
            ),
            "estimated_list_price_usd": (
                "Estimated standard and slim Linux x64 list price before "
                "included minutes, discounts, taxes, or other account "
                "adjustments."
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
            "all_actions_workflow_run_count": len(run_rows),
            "all_actions_workflow_attempt_count": sum(
                row["attempt_count"] for row in run_rows
            ),
            "cumulative_ci_hull_raw_job_minutes": _minutes(
                cumulative_seconds
            ),
            "latest_candidate_ci_hull_raw_job_minutes": _minutes(
                latest_seconds
            ),
            "amplification_ratio": _ratio(
                cumulative_seconds, latest_seconds
            ),
            "package_expansion_raw_job_minutes": _minutes(
                package_expansion_seconds
            ),
            "all_measured_hosted_raw_job_minutes": _minutes(
                all_hosted_seconds
            ),
            "summed_workflow_wall_minutes": _minutes(
                all_workflow_wall_seconds
            ),
            "agent_wait_interval_count": len(wait_rows),
            "agent_wait_minutes": _minutes(all_agent_wait_seconds),
            "untimed_job_count": untimed_jobs,
            "timing_anomaly_count": timing_anomalies,
            "started_job_count": sum(
                int(accounting["started_job_count"])
                for accounting in run_accounting.values()
            ),
            "accounted_job_slot_count": sum(
                int(accounting["started_job_count"])
                + int(accounting["skipped_job_count"])
                + int(accounting["never_started_job_count"])
                + int(accounting["non_vm_synthetic_check_count"])
                for accounting in run_accounting.values()
            ),
            "canceled_started_job_count": sum(
                int(accounting["canceled_started_job_count"])
                for accounting in run_accounting.values()
            ),
            "failed_started_job_count": sum(
                int(accounting["failed_started_job_count"])
                for accounting in run_accounting.values()
            ),
            "skipped_job_count": sum(
                int(accounting["skipped_job_count"])
                for accounting in run_accounting.values()
            ),
            "never_started_job_count": sum(
                int(accounting["never_started_job_count"])
                for accounting in run_accounting.values()
            ),
            "non_vm_synthetic_check_count": sum(
                int(accounting["non_vm_synthetic_check_count"])
                for accounting in run_accounting.values()
            ),
            "linux_x64_started_job_count": sum(
                int(accounting["linux_x64_started_job_count"])
                for accounting in run_accounting.values()
            ),
            "linux_x64_standard_started_job_count": sum(
                int(accounting["linux_x64_standard_started_job_count"])
                for accounting in run_accounting.values()
            ),
            "linux_x64_slim_started_job_count": sum(
                int(accounting["linux_x64_slim_started_job_count"])
                for accounting in run_accounting.values()
            ),
            "self_hosted_started_job_count": sum(
                int(accounting["self_hosted_started_job_count"])
                for accounting in run_accounting.values()
            ),
            "unpriced_started_job_count": unpriced_started_jobs,
            "estimated_billable_linux_x64_minutes": (
                all_estimated_billable_minutes
            ),
            "estimated_billable_linux_x64_standard_minutes": sum(
                int(
                    accounting[
                        "estimated_billable_linux_x64_standard_minutes"
                    ]
                )
                for accounting in run_accounting.values()
            ),
            "estimated_billable_linux_x64_slim_minutes": sum(
                int(
                    accounting["estimated_billable_linux_x64_slim_minutes"]
                )
                for accounting in run_accounting.values()
            ),
            "estimated_list_price_usd": round(
                sum(
                    float(accounting["estimated_list_price_usd"])
                    for accounting in run_accounting.values()
                ),
                6,
            ),
        },
        "pull_requests": pr_rows,
        "workflows": workflow_rows,
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


def _md_cell(value: object) -> str:
    if value is None:
        return "n/a"
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r\n", "<br>")
        .replace("\n", "<br>")
        .replace("\r", "<br>")
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = _mapping(report.get("summary"), "report summary")
    pricing = _mapping(report.get("pricing"), "report pricing")
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
            "| All attributable Actions workflow runs | "
            f"{summary['all_actions_workflow_run_count']} |"
        ),
        f"| Accounted job slots | {summary['accounted_job_slot_count']} |",
        f"| Jobs that started | {summary['started_job_count']} |",
        (
            "| Started jobs canceled | "
            f"{summary['canceled_started_job_count']} |"
        ),
        (
            "| Started jobs failed or timed out | "
            f"{summary['failed_started_job_count']} |"
        ),
        f"| Skipped jobs | {summary['skipped_job_count']} |",
        (
            "| Jobs that never started | "
            f"{summary['never_started_job_count']} |"
        ),
        (
            "| Non-VM synthetic checks | "
            f"{summary['non_vm_synthetic_check_count']} |"
        ),
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
        (
            "| Estimated billable recognized-Linux minutes | "
            f"{summary['estimated_billable_linux_x64_minutes']} |"
        ),
        (
            "| Estimated recognized-Linux list price | $"
            f"{_format_number(summary['estimated_list_price_usd'])} |"
        ),
        "",
        "Raw job-minutes sum job execution and are not billing-rounded runner "
        "minutes. Workflow wall time can overlap across workflows. Agent waiting "
        "comes only from supplied intervals and is neither of those measures. "
        f"The list-price estimate uses ${pricing['linux_x64_minute_rate_usd']}"
        " per standard Linux x64 minute and "
        f"${pricing['linux_x64_slim_minute_rate_usd']} per slim minute, and "
        "rounds every started job separately. "
        "Skipped and never-started jobs are reported but not priced; canceled "
        "and failed jobs that started are priced. Included minutes and account "
        "discounts are not deducted.",
        "",
        "## Pull requests",
        "",
        (
            "| PR | Candidates | Actions runs | Started jobs | Canceled | "
            "Failed | Never started | Non-VM checks | Expansions | "
            "Cumulative CI/Hull job-min | Latest job-min | Amplification | "
            "Agent wait min | Est. billable min | Est. list price |"
        ),
        (
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        ),
    ]
    for row in report["pull_requests"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"#{row['pr_number']}",
                    str(row["candidate_count"]),
                    str(row["actions_workflow_run_count"]),
                    str(row["started_job_count"]),
                    str(row["canceled_started_job_count"]),
                    str(row["failed_started_job_count"]),
                    str(row["never_started_job_count"]),
                    str(row["non_vm_synthetic_check_count"]),
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
                    str(row["estimated_billable_linux_x64_minutes"]),
                    f"${_format_number(row['estimated_list_price_usd'])}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Workflow totals",
            "",
            (
                "| Workflow | Runs | Attempts | Started | Canceled | Failed | "
                "Skipped | Never started | Non-VM checks | Raw job-min | "
                "Est. billable min | Est. list price |"
            ),
            (
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: | ---: |"
            ),
        ]
    )
    for row in report["workflows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(row["workflow"]),
                    str(row["workflow_run_count"]),
                    str(row["workflow_attempt_count"]),
                    str(row["started_job_count"]),
                    str(row["canceled_started_job_count"]),
                    str(row["failed_started_job_count"]),
                    str(row["skipped_job_count"]),
                    str(row["never_started_job_count"]),
                    str(row["non_vm_synthetic_check_count"]),
                    _format_number(row["raw_job_minutes"]),
                    str(row["estimated_billable_linux_x64_minutes"]),
                    f"${_format_number(row['estimated_list_price_usd'])}",
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
                "Agent waits | Agent wait min | Started jobs | Canceled jobs | "
                "Failed jobs | Never started | Non-VM checks | "
                "Est. billable min | Est. list price |"
            ),
            (
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: |"
            ),
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
                    str(row["started_job_count"]),
                    str(row["canceled_started_job_count"]),
                    str(row["failed_started_job_count"]),
                    str(row["never_started_job_count"]),
                    str(row["non_vm_synthetic_check_count"]),
                    str(row["estimated_billable_linux_x64_minutes"]),
                    f"${_format_number(row['estimated_list_price_usd'])}",
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
                "Basis | Runs | Raw job-min | Evidence | Note |"
            ),
            (
                "| ---: | ---: | --- | --- | --- | --- | --- | --- | ---: | "
                "--- | --- |"
            ),
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
                    _md_cell(row["evidence"]),
                    _md_cell(row["note"]),
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
                "| Run | PR | PR basis | Head | Workflow | Created | Cause | "
                "Cause basis | Attempts | Started jobs | Canceled jobs | "
                "Raw job-min | Wall min | Est. billable min | Est. list price | "
                "Evidence | Note |"
            ),
            (
                "| ---: | ---: | --- | --- | --- | --- | --- | --- | ---: | "
                "---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"
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
                    _md_cell(row["pr_attribution"]),
                    f"`{row['head_sha'][:12]}`",
                    _md_cell(row["workflow"]),
                    row["created_at"],
                    row["cause"],
                    row["attribution"],
                    str(row["attempt_count"]),
                    str(row["started_job_count"]),
                    str(row["canceled_started_job_count"]),
                    _format_number(row["raw_job_minutes"]),
                    _format_number(row["workflow_wall_minutes"]),
                    str(row["estimated_billable_linux_x64_minutes"]),
                    f"${_format_number(row['estimated_list_price_usd'])}",
                    _md_cell(row["evidence"]),
                    _md_cell(row["note"]),
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
                    "| Wait ID | Agent | PR | Head | Candidate | Start | End | "
                    "Cause | Run IDs | Wait min | Evidence | Note |"
                ),
                (
                    "| --- | --- | ---: | --- | --- | --- | --- | --- | --- | "
                    "---: | --- | --- |"
                ),
            ]
        )
        for row in report["agent_wait_ledger"]:
            head = (
                f"`{row['head_sha'][:12]}`"
                if row["head_sha"] is not None
                else "n/a"
            )
            candidate = (
                f"`{row['candidate_sha'][:12]}`"
                if row["candidate_sha"] is not None
                else "n/a"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_cell(row["wait_id"]),
                        _md_cell(row["agent_id"]),
                        f"#{row['pr_number']}",
                        head,
                        candidate,
                        row["started_at"],
                        row["ended_at"],
                        row["cause"],
                        ", ".join(str(value) for value in row["run_ids"])
                        or "n/a",
                        _format_number(row["wait_minutes"]),
                        _md_cell(row["evidence"]),
                        _md_cell(row["note"]),
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


def _gh_run_log_text(repository: str, run_id: int) -> str | None:
    completed = subprocess.run(
        ["gh", "api", f"repos/{repository}/actions/runs/{run_id}/logs"],
        capture_output=True,
    )
    if completed.returncode != 0 or not completed.stdout:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(completed.stdout)) as archive:
            parts = []
            for name in sorted(archive.namelist()):
                if name.endswith("/"):
                    continue
                parts.append(
                    archive.read(name).decode("utf-8", errors="replace")
                )
            return "\n".join(parts)
    except (OSError, zipfile.BadZipFile):
        return None


def _logged_parent_run_id(
    workflow_path: str, log_text: str | None
) -> int | None:
    if not log_text:
        return None
    patterns = []
    if workflow_path == ".github/workflows/pr-candidate-receipt.yml":
        patterns.append(r"--trigger-run-id\s+[\"']?(\d+)")
    elif workflow_path == ".github/workflows/openspec-autoland-controller.yml":
        patterns.extend(
            [
                r"--run-id\s+[\"']?(\d+)",
                r"\bRUN_ID:\s*[\"']?(\d+)",
            ]
        )
    matches = {
        int(match)
        for pattern in patterns
        for match in re.findall(pattern, log_text)
    }
    if len(matches) == 1:
        return matches.pop()
    return None


def _workflow_runs(
    api: Api,
    *,
    repository: str,
    since: datetime,
    until: datetime,
) -> list[Mapping[str, Any]]:
    if until < since:
        raise ReportError("Actions collection end precedes its start")

    def collect_interval(
        start: datetime, end: datetime
    ) -> dict[int, Mapping[str, Any]]:
        created_filter = f"{_timestamp_text(start)}..{_timestamp_text(end)}"
        first_query = urlencode(
            {
                "created": created_filter,
                "per_page": 100,
                "page": 1,
            }
        )
        first = _mapping(
            api(f"repos/{repository}/actions/runs?{first_query}"),
            "Actions workflow response",
        )
        total_count = first.get("total_count")
        if type(total_count) is int and total_count > 1000:
            if end - start <= timedelta(seconds=1):
                raise ReportError(
                    "Actions workflow listing exceeds 1,000 runs inside one second"
                )
            midpoint = start + (end - start) / 2
            combined = collect_interval(start, midpoint)
            combined.update(collect_interval(midpoint, end))
            return combined

        collected: dict[int, Mapping[str, Any]] = {}
        for page in range(1, 101):
            if page == 1:
                payload = first
            else:
                query = urlencode(
                    {
                        "created": created_filter,
                        "per_page": 100,
                        "page": page,
                    }
                )
                payload = _mapping(
                    api(f"repos/{repository}/actions/runs?{query}"),
                    "Actions workflow response",
                )
            rows = [
                _mapping(row, "Actions workflow run")
                for row in _sequence(
                    payload.get("workflow_runs"), "Actions workflow_runs"
                )
            ]
            for row in rows:
                run_id = _positive_int(row.get("id"), "workflow run id")
                collected[run_id] = row
            if len(rows) < 100:
                return collected
        raise ReportError("Actions workflow listing exceeded 100 pages")

    return list(collect_interval(since, until).values())


def _pulls_for_head(
    api: Api,
    *,
    repository: str,
    head_repository: str,
    head_ref: str,
) -> list[Mapping[str, Any]]:
    owner = head_repository.split("/", 1)[0]
    query = urlencode(
        {
            "state": "all",
            "head": f"{owner}:{head_ref}",
            "per_page": 100,
        }
    )
    collected: list[Mapping[str, Any]] = []
    for page in range(1, 101):
        payload = _sequence(
            api(f"repos/{repository}/pulls?{query}&page={page}"),
            "pull requests for head",
        )
        rows = [
            _mapping(row, "pull request for head")
            for row in payload
        ]
        collected.extend(rows)
        if len(rows) < 100:
            return collected
    raise ReportError(
        f"pull request listing exceeded 100 pages for {head_repository}:{head_ref}"
    )


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
    if run.get("event") not in {
        "pull_request",
        "pull_request_target",
        "push",
    }:
        return None
    key = _run_head_key(run)
    if key is None:
        return None
    created_at = _timestamp(run.get("created_at"), "workflow run created_at")
    matches = [
        number
        for number, opened_at, closed_at in pull_windows.get(key, [])
        if opened_at <= created_at <= closed_at
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _workflow_run_parent_ids(
    child: Mapping[str, Any],
    *,
    listed_runs: Mapping[int, Mapping[str, Any]],
    run_pr_numbers: Mapping[int, int],
    run_heads: Mapping[int, str],
) -> list[int]:
    parent_names = WORKFLOW_RUN_PARENTS.get(_workflow_path(child))
    if child.get("event") != "workflow_run" or parent_names is None:
        return []
    child_created = _timestamp(
        child.get("created_at"), "workflow_run child created_at"
    )
    candidates: list[tuple[float, int]] = []
    for run_id, run in listed_runs.items():
        if run_id not in run_pr_numbers or run_id not in run_heads:
            continue
        if run.get("name") not in parent_names or run.get("updated_at") is None:
            continue
        parent_completed = _timestamp(
            run.get("updated_at"), f"workflow_run parent {run_id} updated_at"
        )
        delta = (child_created - parent_completed).total_seconds()
        if 0 <= delta <= WORKFLOW_RUN_PARENT_WINDOW_SECONDS:
            candidates.append((delta, run_id))
    if not candidates:
        return []
    cluster = [run_id for _, run_id in candidates]
    pr_numbers = {run_pr_numbers[run_id] for run_id in cluster}
    heads = {run_heads[run_id] for run_id in cluster}
    if len(pr_numbers) != 1 or len(heads) != 1:
        return []
    return sorted(cluster)


def collect_github_data(
    *,
    repository: str,
    prs: Sequence[int],
    attribution_payload: Mapping[str, Any] | None,
    since: datetime,
    api: Api = _gh_api,
    log_api: LogApi | None = None,
) -> dict[str, Any]:
    cohort = set(prs)
    collected_at = datetime.now(timezone.utc)
    ledger = validate_ledger(
        attribution_payload if attribution_payload is not None else _empty_ledger()
    )
    run_attributions = {
        row["run_id"]: row for row in ledger["run_attributions"]
    }
    pulls: list[dict[str, Any]] = []
    cohort_pr_windows: dict[int, tuple[datetime, datetime]] = {}
    cohort_head_keys: set[tuple[str, str]] = set()
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
        closed_at = (
            _timestamp(
                raw_pull.get("closed_at"),
                f"pull request #{pr_number} closed_at",
            )
            if raw_pull.get("closed_at") is not None
            else collected_at
        )
        cohort_pr_windows[pr_number] = (opened_at, closed_at)
        head_key = (head_repository_name, head_ref)
        cohort_head_keys.add(head_key)
        pulls.append(
            {
                "number": pr_number,
                "title": raw_pull.get("title"),
                "state": raw_pull.get("state"),
                "created_at": raw_pull.get("created_at"),
                "updated_at": raw_pull.get("updated_at"),
                "closed_at": raw_pull.get("closed_at"),
                "head_sha": head.get("sha"),
                "head_ref": head_ref,
                "head_repository": head_repository_name,
                "base_ref": base.get("ref"),
            }
        )

    for head_repository_name, head_ref in sorted(cohort_head_keys):
        for raw_pull in _pulls_for_head(
            api,
            repository=repository,
            head_repository=head_repository_name,
            head_ref=head_ref,
        ):
            number = _positive_int(
                raw_pull.get("number"), "branch pull request number"
            )
            head = _mapping(
                raw_pull.get("head"), f"branch pull request #{number} head"
            )
            head_repository = _mapping(
                head.get("repo"),
                f"branch pull request #{number} head repository",
            )
            key = (
                _text(
                    head_repository.get("full_name"),
                    f"branch pull request #{number} head repository name",
                ),
                _text(
                    head.get("ref"),
                    f"branch pull request #{number} head ref",
                ),
            )
            if key != (head_repository_name, head_ref):
                continue
            opened_at = _timestamp(
                raw_pull.get("created_at"),
                f"branch pull request #{number} created_at",
            )
            closed_at = (
                _timestamp(
                    raw_pull.get("closed_at"),
                    f"branch pull request #{number} closed_at",
                )
                if raw_pull.get("closed_at") is not None
                else collected_at
            )
            pull_windows[key].append((number, opened_at, closed_at))

    all_runs: dict[int, Mapping[str, Any]] = {}
    run_pr_numbers: dict[int, int] = {}
    run_pr_bases: dict[int, str] = {}
    run_heads: dict[int, str] = {}
    if log_api is None and api is _gh_api:
        def live_log_api(run_id: int) -> str | None:
            return _gh_run_log_text(repository, run_id)

        log_api = live_log_api
    listed_runs = {
        _positive_int(run.get("id"), "workflow run id"): run
        for run in _workflow_runs(
            api,
            repository=repository,
            since=since,
            until=collected_at,
        )
    }
    for run_id, run in listed_runs.items():
        direct_pr_number = _pull_number(run)
        pr_number = direct_pr_number
        pr_basis = "workflow pull_requests"
        if pr_number is None:
            pr_number = _branch_pull_number(run, pull_windows)
            pr_basis = (
                "exact head repository/ref and non-overlapping PR lifetime"
            )
        attribution = run_attributions.get(run_id)
        if (
            pr_number is not None
            and attribution is not None
            and pr_number != attribution["pr_number"]
        ):
            raise ReportError(
                f"run {run_id} ledger PR disagrees with GitHub identity"
            )
        if direct_pr_number is not None and direct_pr_number not in cohort:
            continue
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
        head_sha = _pull_head(run)
        if (
            head_sha is None
            and isinstance(run.get("head_sha"), str)
            and (
                run_pr_bases[run_id] != "attribution ledger"
                or run.get("event")
                in {"pull_request", "pull_request_target", "push"}
            )
        ):
            head_sha = run["head_sha"]
        if attribution is not None:
            if (
                head_sha is not None
                and run.get("event")
                in {"pull_request", "pull_request_target", "push"}
                and head_sha != attribution["head_sha"]
            ):
                raise ReportError(
                    f"run {run_id} ledger head disagrees with GitHub identity"
                )
            head_sha = attribution["head_sha"]
        if head_sha is not None:
            run_heads[run_id] = head_sha
        all_runs[run_id] = run

    pending_children = [
        run_id
        for run_id, run in listed_runs.items()
        if run_id not in all_runs
        and run.get("event") == "workflow_run"
        and _workflow_path(run) in WORKFLOW_RUN_PARENTS
    ]
    logged_parents: dict[int, int] = {}
    if log_api is not None and pending_children:
        def fetch_logged_parent(run_id: int) -> tuple[int, int | None]:
            path = _workflow_path(listed_runs[run_id])
            return run_id, _logged_parent_run_id(path, log_api(run_id))

        with ThreadPoolExecutor(
            max_workers=min(16, len(pending_children))
        ) as executor:
            for run_id, parent_id in executor.map(
                fetch_logged_parent, sorted(pending_children)
            ):
                if parent_id is not None:
                    logged_parents[run_id] = parent_id

    for run_id, run in listed_runs.items():
        if run_id in all_runs:
            continue
        logged_parent = logged_parents.get(run_id)
        if (
            logged_parent is not None
            and logged_parent in run_pr_numbers
            and logged_parent in run_heads
            and listed_runs.get(logged_parent, {}).get("name")
            in WORKFLOW_RUN_PARENTS.get(_workflow_path(run), set())
        ):
            parent_ids = [logged_parent]
            parent_basis = f"exact logged workflow_run parent {logged_parent}"
        else:
            parent_ids = _workflow_run_parent_ids(
                run,
                listed_runs=listed_runs,
                run_pr_numbers=run_pr_numbers,
                run_heads=run_heads,
            )
            parent_basis = (
                "workflow_run same-PR/head completion cluster "
                + ",".join(map(str, parent_ids))
            )
        if not parent_ids:
            continue
        parent_prs = {run_pr_numbers[parent_id] for parent_id in parent_ids}
        parent_heads = {run_heads[parent_id] for parent_id in parent_ids}
        if len(parent_prs) != 1 or len(parent_heads) != 1:
            continue
        pr_number = parent_prs.pop()
        if pr_number not in cohort:
            continue
        run_pr_numbers[run_id] = pr_number
        run_heads[run_id] = parent_heads.pop()
        run_pr_bases[run_id] = parent_basis
        all_runs[run_id] = dict(run, source_run_ids=parent_ids)

    for run_id, attribution in run_attributions.items():
        if attribution["pr_number"] not in cohort or run_id in all_runs:
            continue
        all_runs[run_id] = _mapping(
            api(f"repos/{repository}/actions/runs/{run_id}"),
            f"attributed workflow run {run_id}",
        )
        run_pr_numbers[run_id] = attribution["pr_number"]
        run_pr_bases[run_id] = "attribution ledger"
        run_heads[run_id] = attribution["head_sha"]

    for run_id, attribution in run_attributions.items():
        if attribution["pr_number"] not in cohort or run_id not in all_runs:
            continue
        run_created = _timestamp(
            all_runs[run_id].get("created_at"),
            f"attributed workflow run {run_id} created_at",
        )
        opened_at, closed_at = cohort_pr_windows[attribution["pr_number"]]
        if run_created < since:
            raise ReportError(
                f"attributed workflow run {run_id} predates --since"
            )
        if not opened_at <= run_created <= closed_at:
            raise ReportError(
                f"attributed workflow run {run_id} falls outside PR "
                f"#{attribution['pr_number']} lifetime"
            )
        raw = all_runs[run_id]
        if raw.get("event") in {
            "pull_request",
            "pull_request_target",
            "push",
        }:
            github_head = _pull_head(raw)
            if github_head is None and isinstance(raw.get("head_sha"), str):
                github_head = raw["head_sha"]
            if github_head is not None and github_head != attribution["head_sha"]:
                raise ReportError(
                    f"run {run_id} ledger head disagrees with GitHub identity"
                )
            if (
                attribution["candidate_sha"] is not None
                and isinstance(raw.get("head_sha"), str)
                and raw["head_sha"] != attribution["candidate_sha"]
            ):
                raise ReportError(
                    f"run {run_id} ledger candidate disagrees with GitHub identity"
                )

    def fetch_jobs(run_id: int) -> tuple[int, list[Mapping[str, Any]]]:
        return run_id, _jobs(api, repository=repository, run_id=run_id)

    jobs_by_run: dict[int, list[Mapping[str, Any]]] = {}
    if all_runs:
        with ThreadPoolExecutor(max_workers=min(16, len(all_runs))) as executor:
            for run_id, jobs in executor.map(fetch_jobs, sorted(all_runs)):
                jobs_by_run[run_id] = jobs

    normalized: list[dict[str, Any]] = []
    for run_id, raw in sorted(
        all_runs.items(),
        key=lambda item: _timestamp(
            item[1].get("created_at"), f"run {item[0]} created_at"
        ),
    ):
        path = _workflow_path(raw)
        attribution = run_attributions.get(run_id)
        pr_number = run_pr_numbers.get(run_id)
        if pr_number not in cohort:
            continue
        head_sha = run_heads.get(run_id) or _pull_head(raw)
        if (
            head_sha is None
            and run_pr_bases[run_id] != "attribution ledger"
            and isinstance(raw.get("head_sha"), str)
        ):
            head_sha = raw["head_sha"]
        if head_sha is None and attribution is not None:
            head_sha = attribution["head_sha"]
        if (
            head_sha is not None
            and attribution is not None
            and raw.get("event")
            in {"pull_request", "pull_request_target", "push"}
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
                "source_run_ids": raw.get("source_run_ids", []),
                "jobs": jobs_by_run[run_id],
            }
        )
    return {
        "schema": GITHUB_SCHEMA,
        "repository": repository,
        "collected_at": _timestamp_text(collected_at),
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
    parser.add_argument(
        "--linux-x64-minute-rate-usd",
        type=float,
        default=DEFAULT_LINUX_X64_RATE_USD,
        help=(
            "standard Linux x64 list price per rounded job-minute "
            f"(default: {DEFAULT_LINUX_X64_RATE_USD})"
        ),
    )
    parser.add_argument(
        "--linux-x64-slim-minute-rate-usd",
        type=float,
        default=DEFAULT_LINUX_X64_SLIM_RATE_USD,
        help=(
            "Linux x64 slim list price per rounded job-minute "
            f"(default: {DEFAULT_LINUX_X64_SLIM_RATE_USD})"
        ),
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
            linux_x64_rate_usd=arguments.linux_x64_minute_rate_usd,
            linux_x64_slim_rate_usd=(
                arguments.linux_x64_slim_minute_rate_usd
            ),
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
        "CI/Hull raw job-minutes, "
        f"${summary['estimated_list_price_usd']:.2f} estimated Linux "
        "list price"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
