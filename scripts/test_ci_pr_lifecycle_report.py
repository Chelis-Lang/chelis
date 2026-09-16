"""Deterministic tests for CI pull-request lifecycle reporting."""

from __future__ import annotations

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
        "conclusion": "success",
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
                "pr_number": 1,
                "head_sha": HEAD_B,
                "cause": "review-repair",
                "started_at": "2026-09-16T11:00:00Z",
                "ended_at": "2026-09-16T11:12:00Z",
                "run_ids": [103, 104],
                "evidence": "trace:review-round-1",
            },
            {
                "wait_id": "expansion-wait",
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
        self.assertIn("No dollar cost is calculated", markdown)

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
                "repos/Chelis-Lang/chelis/actions/workflows/ci.yml/runs"
                "?per_page=100&page=1"
            ): {"workflow_runs": [ci_run]},
            (
                "repos/Chelis-Lang/chelis/actions/workflows/conformance.yml/runs"
                "?per_page=100&page=1"
            ): {"workflow_runs": []},
            (
                "repos/Chelis-Lang/chelis/actions/workflows/"
                "pr-package-expansion.yml/runs"
                "?per_page=100&page=1"
            ): {"workflow_runs": [expansion_run]},
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
            "exact head repository/ref and PR activity window",
        )
        self.assertEqual(by_id[302]["pr_number"], 7)
        self.assertEqual(by_id[302]["head_sha"], HEAD_B)


if __name__ == "__main__":
    unittest.main()
