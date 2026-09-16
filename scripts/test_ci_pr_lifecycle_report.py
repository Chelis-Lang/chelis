"""Deterministic tests for CI pull-request lifecycle reporting."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from scripts import ci_pr_lifecycle_report as report


HEAD_A = "a" * 40
HEAD_B = "b" * 40
HEAD_C = "c" * 40
CANDIDATE_A = "1" * 40
CANDIDATE_B = "2" * 40
CANDIDATE_C = "3" * 40


def job(
    job_id: int,
    *,
    start: str,
    minutes: float,
    attempt: int = 1,
    conclusion: str = "success",
    labels: list[str] | None = None,
) -> dict:
    started = datetime.fromisoformat(start.replace("Z", "+00:00"))
    completed = started.timestamp() + minutes * 60
    return {
        "id": job_id,
        "name": f"job-{job_id}",
        "run_attempt": attempt,
        "started_at": start,
        "completed_at": datetime.fromtimestamp(
            completed, tz=timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "conclusion": conclusion,
        "labels": labels if labels is not None else ["ubuntu-latest"],
    }


def workflow_run(
    run_id: int,
    *,
    pr_number: int,
    head_sha: str,
    candidate_sha: str | None,
    path: str,
    start: str,
    job_minutes: float,
    attempt: int = 1,
) -> dict:
    jobs = [job(run_id * 10, start=start, minutes=job_minutes, attempt=attempt)]
    return {
        "id": run_id,
        "run_attempt": attempt,
        "workflow_path": path,
        "event": (
            "workflow_dispatch"
            if path == report.EXPANSION_WORKFLOW
            else "pull_request"
        ),
        "pr_number": pr_number,
        "head_sha": head_sha,
        "candidate_sha": candidate_sha,
        "created_at": start,
        "run_started_at": start,
        "updated_at": jobs[0]["completed_at"],
        "status": "completed",
        "conclusion": "success",
        "jobs": jobs,
    }


def github_fixture() -> dict:
    payload = {
        "schema": report.GITHUB_SCHEMA,
        "repository": "Chelis-Lang/chelis",
        "pull_requests": [
            {
                "number": 1,
                "title": "First pull request",
                "state": "open",
                "created_at": "2026-09-16T09:00:00Z",
                "updated_at": "2026-09-16T12:00:00Z",
                "head_sha": HEAD_B,
                "base_ref": "main",
            },
            {
                "number": 2,
                "title": "Second pull request",
                "state": "merged",
                "created_at": "2026-09-16T09:30:00Z",
                "updated_at": "2026-09-16T11:00:00Z",
                "head_sha": HEAD_C,
                "base_ref": "main",
            },
        ],
        "workflow_runs": [
            workflow_run(
                101,
                pr_number=1,
                head_sha=HEAD_A,
                candidate_sha=CANDIDATE_A,
                path=".github/workflows/ci.yml",
                start="2026-09-16T10:00:00Z",
                job_minutes=10,
            ),
            workflow_run(
                102,
                pr_number=1,
                head_sha=HEAD_A,
                candidate_sha=CANDIDATE_A,
                path=".github/workflows/conformance.yml",
                start="2026-09-16T10:01:00Z",
                job_minutes=5,
            ),
            workflow_run(
                103,
                pr_number=1,
                head_sha=HEAD_B,
                candidate_sha=CANDIDATE_B,
                path=".github/workflows/ci.yml",
                start="2026-09-16T11:00:00Z",
                job_minutes=8,
            ),
            workflow_run(
                104,
                pr_number=1,
                head_sha=HEAD_B,
                candidate_sha=CANDIDATE_B,
                path=".github/workflows/conformance.yml",
                start="2026-09-16T11:01:00Z",
                job_minutes=2,
            ),
            workflow_run(
                105,
                pr_number=1,
                head_sha=HEAD_B,
                candidate_sha=None,
                path=report.EXPANSION_WORKFLOW,
                start="2026-09-16T11:02:00Z",
                job_minutes=4,
            ),
            workflow_run(
                106,
                pr_number=1,
                head_sha=HEAD_B,
                candidate_sha=None,
                path=report.EXPANSION_WORKFLOW,
                start="2026-09-16T12:00:00Z",
                job_minutes=3,
            ),
            workflow_run(
                201,
                pr_number=2,
                head_sha=HEAD_C,
                candidate_sha=CANDIDATE_C,
                path=".github/workflows/ci.yml",
                start="2026-09-16T10:30:00Z",
                job_minutes=6,
            ),
        ],
    }
    payload["workflow_runs"][0]["jobs"].append(
        {
            "id": 1011,
            "name": "skipped timestamp anomaly",
            "run_attempt": 1,
            "started_at": "2026-09-16T10:00:01Z",
            "completed_at": "2026-09-16T10:00:00Z",
            "conclusion": "skipped",
        }
    )
    return payload


def attribution_fixture() -> dict:
    return {
        "schema": report.LEDGER_SCHEMA,
        "manual_run_scope_complete": True,
        "manual_run_scope_evidence": "audited workflow dispatch list",
        "candidate_attributions": [
            {
                "pr_number": 1,
                "head_sha": HEAD_A,
                "candidate_sha": CANDIDATE_A,
                "cause": "initial-candidate",
                "evidence": "PR opened-event record",
            },
            {
                "pr_number": 1,
                "head_sha": HEAD_B,
                "candidate_sha": CANDIDATE_B,
                "cause": "review-repair",
                "evidence": "trace:review-round-1",
                "note": "one consolidated repair candidate",
            },
            {
                "pr_number": 2,
                "head_sha": HEAD_C,
                "candidate_sha": CANDIDATE_C,
                "cause": "initial-candidate",
                "evidence": "PR opened-event record",
            },
        ],
        "run_attributions": [
            {
                "run_id": 105,
                "pr_number": 1,
                "head_sha": HEAD_B,
                "cause": "ordinary-content-push",
                "evidence": "dispatch record 105",
                "note": "first expansion, not a rerun",
            }
        ],
        "agent_waits": [
            {
                "wait_id": "review-wait",
                "agent_id": "agent-reviewer",
                "pr_number": 1,
                "candidate_sha": CANDIDATE_B,
                "cause": "review-repair",
                "started_at": "2026-09-16T11:00:00Z",
                "ended_at": "2026-09-16T11:12:00Z",
                "run_ids": [103, 104],
                "evidence": "trace:review-round-1",
            },
            {
                "wait_id": "expansion-wait",
                "agent_id": "agent-finalizer",
                "pr_number": 1,
                "head_sha": HEAD_B,
                "cause": "package-expansion-rerun",
                "started_at": "2026-09-16T12:00:00Z",
                "ended_at": "2026-09-16T12:05:00Z",
                "run_ids": [106],
                "evidence": "trace:finalization",
            },
        ],
    }


class LifecycleReportTests(unittest.TestCase):
    def test_latest_cumulative_compute_and_wait_are_separate(self) -> None:
        result = report.build_report(
            github_fixture(),
            prs=[1, 2],
            attribution_payload=attribution_fixture(),
            generated_at="2026-09-16T13:00:00Z",
        )

        summary = result["summary"]
        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["package_expansion_count"], 2)
        self.assertEqual(summary["cumulative_ci_hull_raw_job_minutes"], 31)
        self.assertEqual(summary["latest_candidate_ci_hull_raw_job_minutes"], 16)
        self.assertEqual(summary["amplification_ratio"], 1.938)
        self.assertEqual(summary["package_expansion_raw_job_minutes"], 7)
        self.assertEqual(summary["all_measured_hosted_raw_job_minutes"], 38)
        self.assertEqual(summary["agent_wait_minutes"], 17)

        by_cause = {row["cause"]: row for row in result["causes"]}
        self.assertEqual(by_cause["initial-candidate"]["candidate_count"], 2)
        self.assertEqual(by_cause["initial-candidate"]["ci_hull_raw_job_minutes"], 21)
        self.assertEqual(by_cause["review-repair"]["candidate_count"], 1)
        self.assertEqual(by_cause["review-repair"]["ci_hull_raw_job_minutes"], 10)
        self.assertEqual(by_cause["review-repair"]["agent_wait_minutes"], 12)
        self.assertEqual(
            by_cause["ordinary-content-push"][
                "package_expansion_raw_job_minutes"
            ],
            4,
        )
        self.assertEqual(
            by_cause["package-expansion-rerun"][
                "package_expansion_raw_job_minutes"
            ],
            3,
        )
        self.assertEqual(
            by_cause["package-expansion-rerun"]["agent_wait_minutes"], 5
        )

        markdown = report.render_markdown(result)
        self.assertIn("Raw job-minutes sum job execution", markdown)
        self.assertIn("Agent waiting comes only from supplied intervals", markdown)
        self.assertIn("Estimated standard-Linux list price", markdown)

    def test_unattributed_candidates_are_unknown(self) -> None:
        payload = github_fixture()
        payload["workflow_runs"] = payload["workflow_runs"][:4]
        result = report.build_report(
            payload,
            prs=[1],
            generated_at="2026-09-16T13:00:00Z",
        )

        candidates = result["candidate_ledger"]
        self.assertEqual(candidates[0]["cause"], "unknown")
        self.assertEqual(candidates[0]["attribution"], "unknown")
        self.assertEqual(candidates[1]["cause"], "unknown")
        self.assertEqual(candidates[1]["attribution"], "unknown")
        self.assertEqual(result["summary"]["agent_wait_minutes"], 0)
        self.assertTrue(
            any("No agent wait intervals" in warning for warning in result["warnings"])
        )
        self.assertTrue(
            any(
                "counts are lower bounds" in warning
                for warning in result["warnings"]
            )
        )

    def test_later_expansion_is_inferred_as_a_rerun(self) -> None:
        result = report.build_report(
            github_fixture(),
            prs=[1],
            generated_at="2026-09-16T13:00:00Z",
        )
        expansions = [
            row
            for row in result["run_ledger"]
            if row["workflow_path"] == report.EXPANSION_WORKFLOW
        ]
        self.assertEqual(expansions[0]["cause"], "unknown")
        self.assertEqual(expansions[1]["cause"], "package-expansion-rerun")
        self.assertEqual(expansions[1]["attribution"], "inferred")

    def test_invalid_semantic_cause_and_wait_interval_fail_closed(self) -> None:
        bad_cause = attribution_fixture()
        bad_cause["candidate_attributions"][0]["cause"] = "probably-a-rebase"
        with self.assertRaisesRegex(report.ReportError, "must be one of"):
            report.build_report(
                github_fixture(), prs=[1], attribution_payload=bad_cause
            )

        bad_wait = attribution_fixture()
        bad_wait["agent_waits"][0]["ended_at"] = "2026-09-16T10:59:00Z"
        with self.assertRaisesRegex(report.ReportError, "ends before it starts"):
            report.build_report(
                github_fixture(), prs=[1], attribution_payload=bad_wait
            )

        bad_scope = attribution_fixture()
        del bad_scope["manual_run_scope_evidence"]
        with self.assertRaisesRegex(
            report.ReportError, "manual_run_scope_evidence"
        ):
            report.build_report(
                github_fixture(), prs=[1], attribution_payload=bad_scope
            )

        missing_identity = attribution_fixture()
        del missing_identity["agent_waits"][0]["candidate_sha"]
        with self.assertRaisesRegex(
            report.ReportError, "requires head_sha or candidate_sha"
        ):
            report.build_report(
                github_fixture(),
                prs=[1],
                attribution_payload=missing_identity,
            )

        wrong_candidate = attribution_fixture()
        wrong_candidate["agent_waits"][0]["candidate_sha"] = CANDIDATE_A
        with self.assertRaisesRegex(
            report.ReportError, "does not match its candidate_sha"
        ):
            report.build_report(
                github_fixture(),
                prs=[1],
                attribution_payload=wrong_candidate,
            )

        unknown_run = attribution_fixture()
        unknown_run["agent_waits"][0]["run_ids"] = [999]
        with self.assertRaisesRegex(
            report.ReportError, "references unknown workflow run"
        ):
            report.build_report(
                github_fixture(), prs=[1], attribution_payload=unknown_run
            )

    def test_attempts_and_overlapping_jobs_use_distinct_clocks(self) -> None:
        payload = github_fixture()
        payload["pull_requests"] = payload["pull_requests"][:1]
        measured_run = payload["workflow_runs"][0]
        measured_run["jobs"] = [
            job(
                1010,
                start="2026-09-16T10:00:00Z",
                minutes=10,
                attempt=1,
            ),
            job(
                1011,
                start="2026-09-16T10:02:00Z",
                minutes=5,
                attempt=1,
            ),
            job(
                1012,
                start="2026-09-16T11:00:00Z",
                minutes=3,
                attempt=2,
            ),
        ]
        payload["workflow_runs"] = [measured_run]

        result = report.build_report(
            payload,
            prs=[1],
            generated_at="2026-09-16T13:00:00Z",
        )

        summary = result["summary"]
        self.assertEqual(summary["implementation_workflow_run_count"], 1)
        self.assertEqual(summary["implementation_workflow_attempt_count"], 2)
        self.assertEqual(summary["cumulative_ci_hull_raw_job_minutes"], 18)
        self.assertEqual(summary["summed_workflow_wall_minutes"], 13)

    def test_fractional_seconds_do_not_accumulate_display_rounding(self) -> None:
        payload = github_fixture()
        payload["pull_requests"] = payload["pull_requests"][:1]
        fractional_minutes = 20.02 / 60
        payload["workflow_runs"] = [
            workflow_run(
                401,
                pr_number=1,
                head_sha=HEAD_A,
                candidate_sha=CANDIDATE_A,
                path=".github/workflows/ci.yml",
                start="2026-09-16T10:00:00Z",
                job_minutes=fractional_minutes,
            ),
            workflow_run(
                402,
                pr_number=1,
                head_sha=HEAD_A,
                candidate_sha=CANDIDATE_A,
                path=".github/workflows/conformance.yml",
                start="2026-09-16T10:01:00Z",
                job_minutes=fractional_minutes,
            ),
            workflow_run(
                403,
                pr_number=1,
                head_sha=HEAD_B,
                candidate_sha=CANDIDATE_B,
                path=".github/workflows/ci.yml",
                start="2026-09-16T11:00:00Z",
                job_minutes=fractional_minutes,
            ),
        ]
        ledger = {
            "schema": report.LEDGER_SCHEMA,
            "manual_run_scope_complete": True,
            "manual_run_scope_evidence": "no manual runs in fixture",
            "candidate_attributions": [
                {
                    "pr_number": 1,
                    "head_sha": HEAD_A,
                    "candidate_sha": CANDIDATE_A,
                    "cause": "initial-candidate",
                    "evidence": "fixture opened event",
                },
                {
                    "pr_number": 1,
                    "head_sha": HEAD_B,
                    "candidate_sha": CANDIDATE_B,
                    "cause": "review-repair",
                    "evidence": "fixture repair event",
                },
            ],
            "run_attributions": [],
            "agent_waits": [],
        }

        result = report.build_report(
            payload,
            prs=[1],
            attribution_payload=ledger,
            generated_at="2026-09-16T13:00:00Z",
        )

        summary = result["summary"]
        pr_row = result["pull_requests"][0]
        causes = {row["cause"]: row for row in result["causes"]}
        cause_total = sum(
            row["ci_hull_raw_job_minutes"] for row in result["causes"]
        )
        self.assertEqual(
            summary["cumulative_ci_hull_raw_job_minutes"], 1.001
        )
        self.assertEqual(
            summary["all_measured_hosted_raw_job_minutes"], 1.001
        )
        self.assertEqual(
            pr_row["cumulative_ci_hull_raw_job_minutes"],
            summary["cumulative_ci_hull_raw_job_minutes"],
        )
        self.assertEqual(
            round(cause_total, 3),
            summary["cumulative_ci_hull_raw_job_minutes"],
        )
        self.assertEqual(
            causes["initial-candidate"]["ci_hull_raw_job_minutes"], 0.667
        )
        self.assertEqual(
            causes["review-repair"]["ci_hull_raw_job_minutes"], 0.334
        )
        self.assertEqual(pr_row["amplification_ratio"], 3.0)
        self.assertEqual(summary["amplification_ratio"], 3.0)
        self.assertEqual(
            sum(row["raw_job_minutes"] for row in result["run_ledger"]),
            1.002,
        )

    def test_duplicate_and_stale_ledger_rows_fail_closed(self) -> None:
        duplicate = github_fixture()
        duplicate["workflow_runs"].append(
            copy.deepcopy(duplicate["workflow_runs"][0])
        )
        with self.assertRaisesRegex(report.ReportError, "duplicate workflow run"):
            report.build_report(duplicate, prs=[1, 2])

        rewritten_candidate = attribution_fixture()
        rewritten_candidate["run_attributions"].append(
            {
                "run_id": 101,
                "pr_number": 1,
                "head_sha": HEAD_A,
                "candidate_sha": CANDIDATE_C,
                "cause": "initial-candidate",
                "evidence": "invalid fixture override",
            }
        )
        with self.assertRaisesRegex(
            report.ReportError, "candidate attribution disagrees"
        ):
            report.build_report(
                github_fixture(),
                prs=[1, 2],
                attribution_payload=rewritten_candidate,
            )

        missing_run = attribution_fixture()
        missing_run["run_attributions"].append(
            {
                "run_id": 999,
                "pr_number": 1,
                "head_sha": HEAD_B,
                "cause": "review-repair",
                "evidence": "stale run row",
            }
        )
        with self.assertRaisesRegex(
            report.ReportError, "runs missing from the snapshot"
        ):
            report.build_report(
                github_fixture(),
                prs=[1, 2],
                attribution_payload=missing_run,
            )

        unused_candidate = attribution_fixture()
        unused_candidate["candidate_attributions"].append(
            {
                "pr_number": 1,
                "head_sha": "d" * 40,
                "candidate_sha": "4" * 40,
                "cause": "review-repair",
                "evidence": "stale candidate row",
            }
        )
        with self.assertRaisesRegex(
            report.ReportError, "unused candidate attribution"
        ):
            report.build_report(
                github_fixture(),
                prs=[1, 2],
                attribution_payload=unused_candidate,
            )

    def test_agent_waits_require_runs_and_do_not_overlap_per_agent(self) -> None:
        no_runs = attribution_fixture()
        no_runs["agent_waits"][0]["run_ids"] = []
        with self.assertRaisesRegex(
            report.ReportError, "requires at least one workflow run ID"
        ):
            report.validate_ledger(no_runs)

        overlap = attribution_fixture()
        overlap["agent_waits"][1]["agent_id"] = "agent-reviewer"
        overlap["agent_waits"][1]["started_at"] = "2026-09-16T11:05:00Z"
        overlap["agent_waits"][1]["ended_at"] = "2026-09-16T11:15:00Z"
        with self.assertRaisesRegex(report.ReportError, "overlapping waits"):
            report.validate_ledger(overlap)

    def test_candidate_chronology_uses_workflow_creation_time(self) -> None:
        payload = github_fixture()
        payload["pull_requests"] = payload["pull_requests"][:1]
        first = workflow_run(
            501,
            pr_number=1,
            head_sha=HEAD_A,
            candidate_sha=CANDIDATE_A,
            path=".github/workflows/ci.yml",
            start="2026-09-16T10:00:00Z",
            job_minutes=1,
        )
        first["run_started_at"] = "2026-09-16T12:00:00Z"
        second = workflow_run(
            502,
            pr_number=1,
            head_sha=HEAD_B,
            candidate_sha=CANDIDATE_B,
            path=".github/workflows/ci.yml",
            start="2026-09-16T11:00:00Z",
            job_minutes=1,
        )
        payload["workflow_runs"] = [first, second]
        ledger = attribution_fixture()
        ledger["candidate_attributions"] = ledger["candidate_attributions"][:2]
        ledger["run_attributions"] = []
        ledger["agent_waits"] = []

        result = report.build_report(
            payload,
            prs=[1],
            attribution_payload=ledger,
            generated_at="2026-09-16T13:00:00Z",
        )

        self.assertEqual(
            result["pull_requests"][0]["latest_candidate_sha"], CANDIDATE_B
        )

    def test_attempt_count_includes_skipped_only_rerun(self) -> None:
        payload = github_fixture()
        payload["pull_requests"] = payload["pull_requests"][:1]
        measured = workflow_run(
            601,
            pr_number=1,
            head_sha=HEAD_A,
            candidate_sha=CANDIDATE_A,
            path=".github/workflows/ci.yml",
            start="2026-09-16T10:00:00Z",
            job_minutes=2,
            attempt=1,
        )
        measured["run_attempt"] = 2
        measured["jobs"].append(
            {
                "id": 6011,
                "name": "skipped rerun",
                "run_attempt": 2,
                "started_at": None,
                "completed_at": None,
                "conclusion": "skipped",
                "labels": [],
            }
        )
        payload["workflow_runs"] = [measured]

        result = report.build_report(payload, prs=[1])

        self.assertEqual(
            result["summary"]["implementation_workflow_attempt_count"], 2
        )
        self.assertEqual(result["summary"]["summed_workflow_wall_minutes"], 2)

    def test_inverted_only_jobs_do_not_fall_back_to_run_wall_time(self) -> None:
        payload = github_fixture()
        payload["pull_requests"] = payload["pull_requests"][:1]
        measured = workflow_run(
            602,
            pr_number=1,
            head_sha=HEAD_A,
            candidate_sha=CANDIDATE_A,
            path=".github/workflows/ci.yml",
            start="2026-09-16T10:00:00Z",
            job_minutes=2,
        )
        measured["jobs"] = [
            {
                "id": 6020,
                "name": "inverted",
                "run_attempt": 1,
                "started_at": "2026-09-16T10:02:00Z",
                "completed_at": "2026-09-16T10:01:00Z",
                "conclusion": "failure",
                "labels": ["ubuntu-latest"],
            }
        ]
        payload["workflow_runs"] = [measured]

        result = report.build_report(payload, prs=[1])

        self.assertEqual(result["summary"]["summed_workflow_wall_minutes"], 0)
        self.assertEqual(result["summary"]["all_measured_hosted_raw_job_minutes"], 0)
        self.assertEqual(
            result["summary"]["estimated_billable_linux_x64_minutes"], 1
        )

    def test_all_actions_jobs_and_per_job_rounding_are_accounted(self) -> None:
        payload = github_fixture()
        payload["pull_requests"] = payload["pull_requests"][:1]
        measured = workflow_run(
            701,
            pr_number=1,
            head_sha=HEAD_A,
            candidate_sha=None,
            path=".github/workflows/changelog.yml",
            start="2026-09-16T10:00:00Z",
            job_minutes=0.1,
        )
        measured["workflow_name"] = "Changelog"
        measured["jobs"] = [
            job(
                7010,
                start="2026-09-16T10:00:00Z",
                minutes=0.1,
            ),
            job(
                7011,
                start="2026-09-16T10:01:00Z",
                minutes=1.01,
                conclusion="cancelled",
            ),
            job(
                7012,
                start="2026-09-16T10:03:00Z",
                minutes=2.01,
                conclusion="failure",
            ),
            {
                "id": 7013,
                "name": "skipped",
                "run_attempt": 1,
                "started_at": "2026-09-16T10:00:00Z",
                "completed_at": "2026-09-16T10:00:00Z",
                "conclusion": "skipped",
                "labels": [],
            },
            {
                "id": 7014,
                "name": "never assigned",
                "run_attempt": 1,
                "started_at": None,
                "completed_at": None,
                "conclusion": "cancelled",
                "labels": ["ubuntu-latest"],
            },
        ]
        payload["workflow_runs"] = [measured]

        result = report.build_report(payload, prs=[1])
        summary = result["summary"]

        self.assertEqual(summary["all_actions_workflow_run_count"], 1)
        self.assertEqual(summary["started_job_count"], 3)
        self.assertEqual(summary["canceled_started_job_count"], 1)
        self.assertEqual(summary["failed_started_job_count"], 1)
        self.assertEqual(summary["skipped_job_count"], 1)
        self.assertEqual(summary["never_started_job_count"], 1)
        self.assertEqual(summary["estimated_billable_linux_x64_minutes"], 6)
        self.assertEqual(summary["estimated_list_price_usd"], 0.036)
        self.assertEqual(result["workflows"][0]["workflow"], "Changelog")

    def test_never_started_attempts_and_non_vm_checks_cost_zero(self) -> None:
        payload = github_fixture()
        payload["pull_requests"] = payload["pull_requests"][:1]
        never_started = workflow_run(
            711,
            pr_number=1,
            head_sha=HEAD_A,
            candidate_sha=None,
            path=".github/workflows/pr-candidate-receipt.yml",
            start="2026-09-16T10:00:00Z",
            job_minutes=1,
        )
        never_started["conclusion"] = "cancelled"
        never_started["jobs"] = []
        synthetic = workflow_run(
            712,
            pr_number=1,
            head_sha=HEAD_A,
            candidate_sha=None,
            path=".github/workflows/pr-base-retarget.yml",
            start="2026-09-16T10:01:00Z",
            job_minutes=1,
        )
        synthetic["jobs"] = [
            {
                "id": 7120,
                "name": "synthetic required check",
                "run_attempt": 1,
                "started_at": "2026-09-16T10:01:00Z",
                "completed_at": "2026-09-16T10:02:00Z",
                "conclusion": "failure",
                "labels": [],
                "runner_id": None,
                "runner_name": None,
                "steps": [],
            }
        ]
        payload["workflow_runs"] = [never_started, synthetic]

        result = report.build_report(payload, prs=[1])
        summary = result["summary"]

        self.assertEqual(summary["never_started_job_count"], 1)
        self.assertEqual(summary["non_vm_synthetic_check_count"], 1)
        self.assertEqual(summary["started_job_count"], 0)
        self.assertEqual(summary["all_measured_hosted_raw_job_minutes"], 0)
        self.assertEqual(summary["estimated_billable_linux_x64_minutes"], 0)
        self.assertEqual(summary["estimated_list_price_usd"], 0)

    def test_markdown_escapes_evidence_and_notes(self) -> None:
        result = report.build_report(
            github_fixture(),
            prs=[1, 2],
            attribution_payload=attribution_fixture(),
            generated_at="2026-09-16T13:00:00Z",
        )
        result["candidate_ledger"][0]["evidence"] = "trace|row\nnext"
        result["run_ledger"][0]["note"] = "note|cell\nnext"

        markdown = report.render_markdown(result)

        self.assertIn("trace\\|row<br>next", markdown)
        self.assertIn("note\\|cell<br>next", markdown)

    def test_offline_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            github_path = root / "github.json"
            ledger_path = root / "ledger.json"
            json_output = root / "report.json"
            markdown_output = root / "report.md"
            github_path.write_text(json.dumps(github_fixture()))
            ledger_path.write_text(json.dumps(attribution_fixture()))

            exit_code = report.main(
                [
                    "--prs",
                    "1-2",
                    "--github-input",
                    str(github_path),
                    "--attribution-ledger",
                    str(ledger_path),
                    "--json-output",
                    str(json_output),
                    "--markdown-output",
                    str(markdown_output),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                json.loads(json_output.read_text())["summary"]["candidate_count"],
                3,
            )
            self.assertIn(
                "# CI candidate lifecycle report", markdown_output.read_text()
            )

    def test_parse_prs_accepts_ranges_and_rejects_descending_ranges(self) -> None:
        self.assertEqual(
            report.parse_prs("2105,2106,2111-2115"),
            [2105, 2106, 2111, 2112, 2113, 2114, 2115],
        )
        with self.assertRaisesRegex(report.ReportError, "descending"):
            report.parse_prs("2115-2111")

    def test_live_collection_uses_mocked_github_data(self) -> None:
        ledger = {
            "schema": report.LEDGER_SCHEMA,
            "manual_run_scope_complete": True,
            "manual_run_scope_evidence": "audited workflow dispatch list",
            "candidate_attributions": [],
            "run_attributions": [
                {
                    "run_id": 302,
                    "pr_number": 7,
                    "head_sha": HEAD_B,
                    "cause": "package-expansion-rerun",
                    "evidence": "dispatch record 302",
                }
            ],
            "agent_waits": [],
        }
        ci_run = {
            "id": 301,
            "run_attempt": 1,
            "path": ".github/workflows/ci.yml",
            "name": "CI",
            "event": "pull_request",
            "head_sha": HEAD_A,
            "created_at": "2026-09-16T10:00:00Z",
            "run_started_at": "2026-09-16T10:00:00Z",
            "updated_at": "2026-09-16T10:02:00Z",
            "head_branch": "agent/measured-pr",
            "head_repository": {"full_name": "Chelis-Lang/chelis"},
            "pull_requests": [],
        }
        expansion_run = {
            "id": 302,
            "run_attempt": 1,
            "path": report.EXPANSION_WORKFLOW,
            "name": "PR Package Expansion",
            "event": "workflow_dispatch",
            "head_sha": HEAD_C,
            "created_at": "2026-09-16T11:00:00Z",
            "run_started_at": "2026-09-16T11:00:00Z",
            "updated_at": "2026-09-16T11:03:00Z",
            "pull_requests": [],
        }
        responses = {
            "repos/Chelis-Lang/chelis/pulls/7": {
                "number": 7,
                "title": "Measured pull request",
                "state": "open",
                "created_at": "2026-09-16T09:00:00Z",
                "updated_at": "2026-09-16T11:03:00Z",
                "head": {
                    "sha": HEAD_B,
                    "ref": "agent/measured-pr",
                    "repo": {"full_name": "Chelis-Lang/chelis"},
                },
                "base": {"ref": "main"},
            },
            (
                "repos/Chelis-Lang/chelis/pulls?state=all&"
                "head=Chelis-Lang%3Aagent%2Fmeasured-pr&per_page=100&page=1"
            ): [
                {
                    "number": 7,
                    "created_at": "2026-09-16T09:00:00Z",
                    "closed_at": None,
                    "head": {
                        "ref": "agent/measured-pr",
                        "repo": {"full_name": "Chelis-Lang/chelis"},
                    },
                }
            ],
            (
                "repos/Chelis-Lang/chelis/actions/runs/301/jobs"
                "?filter=all&per_page=100&page=1"
            ): {
                "jobs": [
                    job(
                        3010,
                        start="2026-09-16T10:00:00Z",
                        minutes=2,
                    )
                ]
            },
            (
                "repos/Chelis-Lang/chelis/actions/runs/302/jobs"
                "?filter=all&per_page=100&page=1"
            ): {
                "jobs": [
                    job(
                        3020,
                        start="2026-09-16T11:00:00Z",
                        minutes=3,
                    )
                ]
            },
        }

        def fake_api(endpoint: str) -> object:
            if endpoint.startswith(
                "repos/Chelis-Lang/chelis/actions/runs?created="
            ):
                return {
                    "total_count": 2,
                    "workflow_runs": [ci_run, expansion_run],
                }
            self.assertIn(endpoint, responses)
            return responses[endpoint]

        payload = report.collect_github_data(
            repository="Chelis-Lang/chelis",
            prs=[7],
            attribution_payload=ledger,
            since=datetime(2026, 9, 16, tzinfo=timezone.utc),
            api=fake_api,
        )

        self.assertEqual(len(payload["workflow_runs"]), 2)
        by_id = {row["id"]: row for row in payload["workflow_runs"]}
        self.assertEqual(by_id[301]["head_sha"], HEAD_A)
        self.assertEqual(by_id[301]["candidate_sha"], HEAD_A)
        self.assertEqual(
            by_id[301]["pr_attribution"],
            "exact head repository/ref and non-overlapping PR lifetime",
        )
        self.assertEqual(by_id[302]["pr_number"], 7)
        self.assertEqual(by_id[302]["head_sha"], HEAD_B)

    def test_live_collection_does_not_reassign_a_reused_branch(self) -> None:
        old_run = {
            "id": 801,
            "run_attempt": 1,
            "path": ".github/workflows/changelog.yml",
            "name": "Changelog",
            "event": "pull_request",
            "head_sha": HEAD_A,
            "created_at": "2026-09-16T10:00:00Z",
            "run_started_at": "2026-09-16T10:00:00Z",
            "updated_at": "2026-09-16T10:01:00Z",
            "head_branch": "agent/reused",
            "head_repository": {"full_name": "Chelis-Lang/chelis"},
            "pull_requests": [],
        }
        current_run = {
            **old_run,
            "id": 802,
            "head_sha": HEAD_B,
            "created_at": "2026-09-16T13:00:00Z",
            "run_started_at": "2026-09-16T13:00:00Z",
            "updated_at": "2026-09-16T13:01:00Z",
        }
        branch_pulls = [
            {
                "number": 6,
                "created_at": "2026-09-16T09:00:00Z",
                "closed_at": "2026-09-16T11:00:00Z",
                "head": {
                    "ref": "agent/reused",
                    "repo": {"full_name": "Chelis-Lang/chelis"},
                },
            },
            {
                "number": 7,
                "created_at": "2026-09-16T12:00:00Z",
                "closed_at": None,
                "head": {
                    "ref": "agent/reused",
                    "repo": {"full_name": "Chelis-Lang/chelis"},
                },
            },
        ]
        responses = {
            "repos/Chelis-Lang/chelis/pulls/7": {
                "number": 7,
                "title": "Current use of branch",
                "state": "open",
                "created_at": "2026-09-16T12:00:00Z",
                "updated_at": "2026-09-16T13:01:00Z",
                "closed_at": None,
                "head": {
                    "sha": HEAD_B,
                    "ref": "agent/reused",
                    "repo": {"full_name": "Chelis-Lang/chelis"},
                },
                "base": {"ref": "main"},
            },
            (
                "repos/Chelis-Lang/chelis/pulls?state=all&"
                "head=Chelis-Lang%3Aagent%2Freused&per_page=100&page=1"
            ): branch_pulls,
            (
                "repos/Chelis-Lang/chelis/actions/runs/802/jobs"
                "?filter=all&per_page=100&page=1"
            ): {
                "jobs": [
                    job(
                        8020,
                        start="2026-09-16T13:00:00Z",
                        minutes=1,
                    )
                ]
            },
        }

        def fake_api(endpoint: str) -> object:
            if endpoint.startswith(
                "repos/Chelis-Lang/chelis/actions/runs?created="
            ):
                return {
                    "total_count": 2,
                    "workflow_runs": [current_run, old_run],
                }
            self.assertIn(endpoint, responses)
            return responses[endpoint]

        payload = report.collect_github_data(
            repository="Chelis-Lang/chelis",
            prs=[7],
            attribution_payload=None,
            since=datetime(2026, 9, 16, tzinfo=timezone.utc),
            api=fake_api,
        )

        self.assertEqual(
            [run["id"] for run in payload["workflow_runs"]], [802]
        )

    def test_live_collection_attributes_pull_request_target_by_branch_lifetime(
        self,
    ) -> None:
        run = {
            "id": 851,
            "run_attempt": 1,
            "path": ".github/workflows/pr-base-retarget.yml",
            "name": "PR Base Retarget Validation",
            "event": "pull_request_target",
            "head_sha": "f" * 40,
            "created_at": "2026-09-16T13:00:00Z",
            "run_started_at": "2026-09-16T13:00:00Z",
            "updated_at": "2026-09-16T13:01:00Z",
            "head_branch": "agent/target-event",
            "head_repository": {"full_name": "Chelis-Lang/chelis"},
            "pull_requests": [],
        }
        responses = {
            "repos/Chelis-Lang/chelis/pulls/7": {
                "number": 7,
                "title": "Target event",
                "state": "open",
                "created_at": "2026-09-16T12:00:00Z",
                "updated_at": "2026-09-16T13:01:00Z",
                "closed_at": None,
                "head": {
                    "sha": HEAD_B,
                    "ref": "agent/target-event",
                    "repo": {"full_name": "Chelis-Lang/chelis"},
                },
                "base": {"ref": "main"},
            },
            (
                "repos/Chelis-Lang/chelis/pulls?state=all&"
                "head=Chelis-Lang%3Aagent%2Ftarget-event&per_page=100&page=1"
            ): [
                {
                    "number": 7,
                    "created_at": "2026-09-16T12:00:00Z",
                    "closed_at": None,
                    "head": {
                        "ref": "agent/target-event",
                        "repo": {"full_name": "Chelis-Lang/chelis"},
                    },
                }
            ],
            (
                "repos/Chelis-Lang/chelis/actions/runs/851/jobs"
                "?filter=all&per_page=100&page=1"
            ): {
                "jobs": [
                    job(
                        8510,
                        start="2026-09-16T13:00:00Z",
                        minutes=1,
                    )
                ]
            },
        }

        def fake_api(endpoint: str) -> object:
            if endpoint.startswith(
                "repos/Chelis-Lang/chelis/actions/runs?created="
            ):
                return {"total_count": 1, "workflow_runs": [run]}
            self.assertIn(endpoint, responses)
            return responses[endpoint]

        payload = report.collect_github_data(
            repository="Chelis-Lang/chelis",
            prs=[7],
            attribution_payload=None,
            since=datetime(2026, 9, 16, tzinfo=timezone.utc),
            api=fake_api,
        )

        attributed = payload["workflow_runs"][0]
        self.assertEqual(attributed["pr_number"], 7)
        self.assertEqual(attributed["head_sha"], HEAD_B)
        self.assertEqual(
            attributed["pr_attribution"],
            "exact head repository/ref and non-overlapping PR lifetime",
        )

    def test_live_collection_attributes_workflow_run_children_to_parents(
        self,
    ) -> None:
        parent = {
            "id": 901,
            "run_attempt": 1,
            "path": ".github/workflows/ci.yml",
            "name": "CI",
            "event": "pull_request",
            "head_sha": CANDIDATE_A,
            "created_at": "2026-09-16T10:00:00Z",
            "run_started_at": "2026-09-16T10:00:00Z",
            "updated_at": "2026-09-16T10:02:00Z",
            "head_branch": "agent/child-source",
            "head_repository": {"full_name": "Chelis-Lang/chelis"},
            "pull_requests": [
                {"number": 7, "head": {"sha": HEAD_A}}
            ],
        }
        child = {
            "id": 902,
            "run_attempt": 1,
            "path": ".github/workflows/pr-candidate-receipt.yml",
            "name": "PR Candidate Receipt",
            "event": "workflow_run",
            "head_sha": "f" * 40,
            "created_at": "2026-09-16T10:02:02Z",
            "run_started_at": "2026-09-16T10:02:03Z",
            "updated_at": "2026-09-16T10:03:00Z",
            "head_branch": "main",
            "head_repository": {"full_name": "Chelis-Lang/chelis"},
            "pull_requests": [],
        }
        responses = {
            "repos/Chelis-Lang/chelis/pulls/7": {
                "number": 7,
                "title": "Parent and child",
                "state": "open",
                "created_at": "2026-09-16T09:00:00Z",
                "updated_at": "2026-09-16T10:03:00Z",
                "closed_at": None,
                "head": {
                    "sha": HEAD_A,
                    "ref": "agent/child-source",
                    "repo": {"full_name": "Chelis-Lang/chelis"},
                },
                "base": {"ref": "main"},
            },
            (
                "repos/Chelis-Lang/chelis/pulls?state=all&"
                "head=Chelis-Lang%3Aagent%2Fchild-source&per_page=100&page=1"
            ): [
                {
                    "number": 7,
                    "created_at": "2026-09-16T09:00:00Z",
                    "closed_at": None,
                    "head": {
                        "ref": "agent/child-source",
                        "repo": {"full_name": "Chelis-Lang/chelis"},
                    },
                }
            ],
            (
                "repos/Chelis-Lang/chelis/actions/runs/901/jobs"
                "?filter=all&per_page=100&page=1"
            ): {
                "jobs": [
                    job(
                        9010,
                        start="2026-09-16T10:00:00Z",
                        minutes=2,
                    )
                ]
            },
            (
                "repos/Chelis-Lang/chelis/actions/runs/902/jobs"
                "?filter=all&per_page=100&page=1"
            ): {
                "jobs": [
                    job(
                        9020,
                        start="2026-09-16T10:02:03Z",
                        minutes=0.5,
                    )
                ]
            },
        }

        def fake_api(endpoint: str) -> object:
            if endpoint.startswith(
                "repos/Chelis-Lang/chelis/actions/runs?created="
            ):
                return {
                    "total_count": 2,
                    "workflow_runs": [child, parent],
                }
            self.assertIn(endpoint, responses)
            return responses[endpoint]

        payload = report.collect_github_data(
            repository="Chelis-Lang/chelis",
            prs=[7],
            attribution_payload=None,
            since=datetime(2026, 9, 16, tzinfo=timezone.utc),
            api=fake_api,
        )
        by_id = {run["id"]: run for run in payload["workflow_runs"]}

        self.assertEqual(by_id[902]["pr_number"], 7)
        self.assertEqual(by_id[902]["head_sha"], HEAD_A)
        self.assertEqual(by_id[902]["source_run_ids"], [901])

        result = report.build_report(payload, prs=[7])
        child_row = next(
            row for row in result["run_ledger"] if row["run_id"] == 902
        )
        self.assertEqual(child_row["cause"], "unknown")
        self.assertEqual(
            child_row["attribution"], "inferred-workflow-run-parent"
        )

    def test_workflow_run_fallback_requires_one_same_pr_head_cluster(
        self,
    ) -> None:
        child = {
            "event": "workflow_run",
            "path": ".github/workflows/pr-candidate-receipt.yml",
            "created_at": "2026-09-16T10:02:15Z",
        }
        listed_runs = {
            901: {
                "name": "CI",
                "updated_at": "2026-09-16T10:02:00Z",
            },
            902: {
                "name": "Hull Conformance",
                "updated_at": "2026-09-16T10:02:10Z",
            },
        }

        self.assertEqual(
            report._workflow_run_parent_ids(
                child,
                listed_runs=listed_runs,
                run_pr_numbers={901: 7, 902: 7},
                run_heads={901: HEAD_A, 902: HEAD_A},
            ),
            [901, 902],
        )
        self.assertEqual(
            report._workflow_run_parent_ids(
                child,
                listed_runs=listed_runs,
                run_pr_numbers={901: 7, 902: 8},
                run_heads={901: HEAD_A, 902: HEAD_A},
            ),
            [],
        )
        child["created_at"] = "2026-09-16T10:02:16Z"
        self.assertEqual(
            report._workflow_run_parent_ids(
                child,
                listed_runs=listed_runs,
                run_pr_numbers={901: 7, 902: 7},
                run_heads={901: HEAD_A, 902: HEAD_A},
            ),
            [902],
        )

    def test_actions_listing_splits_windows_over_github_limit(self) -> None:
        calls = 0
        left = workflow_run(
            951,
            pr_number=7,
            head_sha=HEAD_A,
            candidate_sha=CANDIDATE_A,
            path=".github/workflows/ci.yml",
            start="2026-09-16T10:00:00Z",
            job_minutes=1,
        )
        right = workflow_run(
            952,
            pr_number=7,
            head_sha=HEAD_B,
            candidate_sha=CANDIDATE_B,
            path=".github/workflows/ci.yml",
            start="2026-09-16T11:00:00Z",
            job_minutes=1,
        )

        def fake_api(endpoint: str) -> object:
            nonlocal calls
            self.assertIn("/actions/runs?created=", endpoint)
            calls += 1
            if calls == 1:
                return {"total_count": 1001, "workflow_runs": []}
            if calls == 2:
                return {"total_count": 1, "workflow_runs": [left]}
            return {"total_count": 1, "workflow_runs": [right]}

        runs = report._workflow_runs(
            fake_api,
            repository="Chelis-Lang/chelis",
            since=datetime(2026, 9, 16, 9, tzinfo=timezone.utc),
            until=datetime(2026, 9, 16, 12, tzinfo=timezone.utc),
        )

        self.assertEqual({run["id"] for run in runs}, {951, 952})
        self.assertEqual(calls, 3)


if __name__ == "__main__":
    unittest.main()
