from __future__ import annotations

import copy
from io import BytesIO
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

from scripts import ci_candidate_identity as identity
from scripts import ci_candidate_receipt as receipt
from scripts import ci_contract_paths
from scripts.test_ci_candidate_identity import CandidateRepository, git


def artifact(payload: dict[str, object]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            receipt.IDENTITY_FILENAME,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
    return buffer.getvalue()


class FakeApi:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def __call__(self, endpoint: str) -> object:
        self.calls.append(endpoint)
        if endpoint not in self.responses:
            raise AssertionError(f"unexpected API request: {endpoint}")
        return copy.deepcopy(self.responses[endpoint])


class ReceiptFixture:
    def __init__(self) -> None:
        self.repository = CandidateRepository()
        self.repo_name = "Chelis-Lang/chelis"
        self.pr_number = 2093
        self.app_id = 15368
        self.run_ids = {
            "ci.yml": 101,
            "conformance.yml": 102,
            "changelog.yml": 103,
            "pr-contract-acknowledgements.yml": 104,
            "pr-base-retarget.yml": 105,
            "secret-scan.yml": 107,
        }
        self.contexts = {
            "ci.yml": [
                "Lint and Unit Tests (Linux)",
                "Integration Tests (Linux)",
                "Backend Sanitizers",
                "SMT Feature Build (Linux)",
                "Docs",
                "No AI authorship markers",
            ],
            "conformance.yml": ["Hull Conformance Gate (Linux)"],
            "changelog.yml": ["Changelog"],
            "pr-contract-acknowledgements.yml": [
                "PR Contract Acknowledgements"
            ],
            "pr-base-retarget.yml": ["PR Base Retarget Validation"],
            "secret-scan.yml": ["Secret scan"],
        }
        self.job_ids: dict[str, int] = {}
        next_job = 1001
        required = []
        check_runs = []
        responses: dict[str, object] = {
            f"repos/{self.repo_name}/pulls/{self.pr_number}": {
                "number": self.pr_number,
                "state": "open",
                "head": {
                    "sha": self.repository.head_sha,
                    "repo": {"full_name": self.repo_name},
                },
                "base": {
                    "ref": "main",
                    "repo": {"full_name": self.repo_name},
                },
            },
            f"repos/{self.repo_name}/branches/main": {
                "protected": True,
                "protection": {
                    "required_status_checks": {"checks": required}
                },
            },
            (
                f"repos/{self.repo_name}/commits/{self.repository.head_sha}/"
                "check-runs?per_page=100&page=1"
            ): {"check_runs": check_runs},
            (
                f"repos/{self.repo_name}/commits/{self.repository.head_sha}/"
                "status?per_page=100&page=1"
            ): {"statuses": []},
        }
        for workflow_file, contexts in self.contexts.items():
            run_id = self.run_ids[workflow_file]
            run = self.run(workflow_file)
            responses[
                f"repos/{self.repo_name}/actions/runs/{run_id}"
            ] = run
            for context in contexts:
                job_id = next_job
                next_job += 1
                self.job_ids[context] = job_id
                responses[
                    f"repos/{self.repo_name}/actions/jobs/{job_id}"
                ] = {
                    "id": job_id,
                    "run_id": run_id,
                    "run_attempt": 1,
                    "head_sha": self.repository.head_sha,
                    "name": context,
                    "status": "completed",
                    "conclusion": "success",
                }
                required.append({"context": context, "app_id": self.app_id})
                check_runs.append(
                    {
                        "id": job_id,
                        "name": context,
                        "status": "completed",
                        "conclusion": "success",
                        "started_at": "2026-09-15T12:00:00Z",
                        "app": {"id": self.app_id},
                    }
                )

        self.identities: dict[str, dict[str, object]] = {}
        self.artifacts: dict[int, bytes] = {}
        for index, workflow_file in enumerate(("ci.yml", "conformance.yml"), 1):
            run_id = self.run_ids[workflow_file]
            payload = identity.build_identity(
                repository_path=self.repository.path,
                repository=self.repo_name,
                workflow_file=workflow_file,
                run_id=run_id,
                run_attempt=1,
                pr_number=self.pr_number,
                head_sha=self.repository.head_sha,
                base_ref="main",
                base_sha=self.repository.base_sha,
                candidate_sha=self.repository.candidate_sha,
            )
            self.identities[workflow_file] = payload
            artifact_id = 500 + index
            responses[
                f"repos/{self.repo_name}/actions/runs/{run_id}/artifacts"
                "?per_page=100&page=1"
            ] = {
                "artifacts": [
                    {
                        "id": artifact_id,
                        "name": receipt.WORKFLOWS[workflow_file].identity_artifact,
                        "expired": False,
                    }
                ]
            }
            self.artifacts[artifact_id] = artifact(payload)

        self.responses = responses
        self.trigger_run_id = self.run_ids["pr-contract-acknowledgements.yml"]

    def close(self) -> None:
        self.repository.close()

    def run(self, workflow_file: str) -> dict[str, object]:
        return {
            "id": self.run_ids[workflow_file],
            "run_attempt": 1,
            "event": "pull_request",
            "path": f".github/workflows/{workflow_file}",
            "head_sha": self.repository.head_sha,
            "head_repository": {"full_name": self.repo_name},
            "status": "completed",
            "conclusion": "success",
            "pull_requests": [{"number": self.pr_number}],
        }

    def download(self, artifact_id: int) -> bytes:
        if artifact_id not in self.artifacts:
            raise AssertionError(f"unexpected artifact download: {artifact_id}")
        return self.artifacts[artifact_id]


class CandidateReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ReceiptFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def collect(
        self,
        *,
        responses: dict[str, object] | None = None,
        artifacts: dict[int, bytes] | None = None,
    ) -> dict[str, object]:
        api = FakeApi(responses or self.fixture.responses)
        available = artifacts or self.fixture.artifacts

        def download(artifact_id: int) -> bytes:
            return available[artifact_id]

        return receipt.collect_receipt(
            repository=self.fixture.repo_name,
            trigger_run_id=self.fixture.trigger_run_id,
            receipt_run_id=9001,
            repository_path=self.fixture.repository.path,
            api=api,
            download_artifact=download,
        )

    def test_issues_trusted_receipt_for_same_candidate_and_required_evidence(
        self,
    ) -> None:
        result = self.collect()

        self.assertEqual(result["schema"], receipt.SCHEMA)
        self.assertEqual(result["pr_number"], self.fixture.pr_number)
        self.assertEqual(result["head_sha"], self.fixture.repository.head_sha)
        self.assertEqual(
            result["candidate_sha"], self.fixture.repository.candidate_sha
        )
        self.assertEqual(
            result["candidate_parents"],
            [self.fixture.repository.base_sha, self.fixture.repository.head_sha],
        )
        self.assertEqual(result["patch_id"], self.fixture.identities["ci.yml"]["patch_id"])
        self.assertEqual(
            result["patch_digest"],
            self.fixture.identities["ci.yml"]["patch_digest"],
        )
        self.assertEqual(
            {entry["context"] for entry in result["required_checks"]},
            set(self.fixture.job_ids),
        )
        by_context = {
            entry["context"]: entry for entry in result["required_checks"]
        }
        for context, job_id in self.fixture.job_ids.items():
            self.assertEqual(by_context[context]["check_run_id"], job_id)
            self.assertEqual(by_context[context]["app_id"], self.fixture.app_id)
        self.assertEqual(result["receipt_workflow_run_id"], 9001)
        self.assertEqual(
            set(result["workflow_runs"]),
            set(self.fixture.run_ids),
        )
        self.assertTrue(result["reuse_eligible"])
        self.assertEqual(result["reuse_blockers"], [])

    def test_ci_contract_patch_is_recorded_but_not_reuse_eligible(self) -> None:
        eligible, blockers = receipt.reuse_eligibility(
            [
                "crates/chelis-types/src/lib.rs",
                ".github/workflows/ci.yml",
                "scripts/changelog.py",
                "scripts/phase3_test_change_report.py",
                "scripts/test_ci_candidate_receipt.py",
                "scripts/test_gate.py",
            ]
        )

        self.assertFalse(eligible)
        self.assertEqual(
            blockers,
            [
                ".github/workflows/ci.yml",
                "scripts/changelog.py",
                "scripts/phase3_test_change_report.py",
                "scripts/test_ci_candidate_receipt.py",
                "scripts/test_gate.py",
            ],
        )
        self.assertEqual(
            receipt.reuse_eligibility(
                list(ci_contract_paths.CI_CONTRACT_EXACT_PATHS)
            ),
            ci_contract_paths.reuse_eligibility(
                list(ci_contract_paths.CI_CONTRACT_EXACT_PATHS)
            ),
        )

    def test_renamed_ci_script_records_both_paths_and_blocks_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            git(repository, "init", "-b", "main")
            git(repository, "config", "user.name", "CI Test")
            git(repository, "config", "user.email", "ci@example.invalid")
            script = repository / "scripts" / "ci_guard.py"
            script.parent.mkdir()
            script.write_text("print('guard')\n")
            git(repository, "add", "scripts/ci_guard.py")
            git(repository, "commit", "-m", "root")

            git(repository, "switch", "-c", "feature")
            (repository / "notes").mkdir()
            git(
                repository,
                "mv",
                "scripts/ci_guard.py",
                "notes/guard.py",
            )
            git(repository, "commit", "-m", "move guard")
            head_sha = git(repository, "rev-parse", "HEAD")

            git(repository, "switch", "main")
            (repository / "base.txt").write_text("base\n")
            git(repository, "add", "base.txt")
            git(repository, "commit", "-m", "advance base")
            base_sha = git(repository, "rev-parse", "HEAD")
            git(
                repository,
                "merge",
                "--no-ff",
                "feature",
                "-m",
                "synthetic candidate",
            )
            candidate_sha = git(repository, "rev-parse", "HEAD")

            candidate = identity.build_identity(
                repository_path=repository,
                repository="Chelis-Lang/chelis",
                workflow_file="ci.yml",
                run_id=101,
                run_attempt=1,
                pr_number=2098,
                head_sha=head_sha,
                base_ref="main",
                base_sha=base_sha,
                candidate_sha=candidate_sha,
            )

            self.assertEqual(
                candidate["changed_paths"],
                ["notes/guard.py", "scripts/ci_guard.py"],
            )
            self.assertEqual(
                receipt.reuse_eligibility(candidate["changed_paths"]),
                (False, ["scripts/ci_guard.py"]),
            )

    def test_withholds_receipt_when_latest_required_check_is_not_green(self) -> None:
        responses = copy.deepcopy(self.fixture.responses)
        endpoint = (
            f"repos/{self.fixture.repo_name}/commits/"
            f"{self.fixture.repository.head_sha}/check-runs?per_page=100&page=1"
        )
        check = next(
            item
            for item in responses[endpoint]["check_runs"]
            if item["name"] == "Docs"
        )
        check["status"] = "in_progress"
        check["conclusion"] = None

        with self.assertRaisesRegex(receipt.ReceiptNotReady, "Docs"):
            self.collect(responses=responses)

    def test_withholds_receipt_when_pull_request_head_has_advanced(self) -> None:
        responses = copy.deepcopy(self.fixture.responses)
        pull = responses[
            f"repos/{self.fixture.repo_name}/pulls/{self.fixture.pr_number}"
        ]
        pull["head"]["sha"] = "f" * 40

        with self.assertRaisesRegex(receipt.ReceiptNotReady, "head advanced"):
            self.collect(responses=responses)

    def test_rejects_required_job_from_the_wrong_workflow_path(self) -> None:
        responses = copy.deepcopy(self.fixture.responses)
        run = responses[
            f"repos/{self.fixture.repo_name}/actions/runs/"
            f"{self.fixture.run_ids['ci.yml']}"
        ]
        run["path"] = ".github/workflows/untrusted.yml"

        with self.assertRaisesRegex(receipt.ReceiptError, "run path"):
            self.collect(responses=responses)

    def test_accepts_metadata_edit_skipped_retarget_job_from_trusted_workflow(
        self,
    ) -> None:
        responses = copy.deepcopy(self.fixture.responses)
        endpoint = (
            f"repos/{self.fixture.repo_name}/commits/"
            f"{self.fixture.repository.head_sha}/check-runs?per_page=100&page=1"
        )
        check = next(
            item
            for item in responses[endpoint]["check_runs"]
            if item["name"] == "PR Base Retarget Validation"
        )
        check["id"] = 2001
        check["conclusion"] = "skipped"
        check["started_at"] = "2026-09-15T12:01:00Z"
        responses[
            f"repos/{self.fixture.repo_name}/actions/jobs/2001"
        ] = {
            "id": 2001,
            "run_id": 106,
            "run_attempt": 1,
            "head_sha": self.fixture.repository.head_sha,
            "name": "PR Base Retarget Validation",
            "status": "completed",
            "conclusion": "skipped",
        }
        responses[
            f"repos/{self.fixture.repo_name}/actions/runs/106"
        ] = {
            "id": 106,
            "run_attempt": 1,
            "event": "pull_request_target",
            "path": ".github/workflows/pr-base-retarget.yml",
            "head_sha": self.fixture.repository.head_sha,
            "head_repository": {"full_name": self.fixture.repo_name},
            "status": "completed",
            "conclusion": "skipped",
            "pull_requests": [{"number": self.fixture.pr_number}],
        }

        result = self.collect(responses=responses)

        self.assertEqual(
            result["workflow_runs"]["pr-base-retarget.yml"]["id"], 106
        )

    def secret_scan_push_run(
        self,
        responses: dict[str, object],
        *,
        conclusion: str = "success",
        started_at: str = "2026-09-15T12:05:00Z",
    ) -> None:
        """Add the push-event Secret scan run every pushed head also carries."""
        endpoint = (
            f"repos/{self.fixture.repo_name}/commits/"
            f"{self.fixture.repository.head_sha}/check-runs?per_page=100&page=1"
        )
        responses[endpoint]["check_runs"].append(
            {
                "id": 3001,
                "name": "Secret scan",
                "status": "completed",
                "conclusion": conclusion,
                "started_at": started_at,
                "app": {"id": self.fixture.app_id},
            }
        )
        responses[f"repos/{self.fixture.repo_name}/actions/jobs/3001"] = {
            "id": 3001,
            "run_id": 108,
            "run_attempt": 1,
            "head_sha": self.fixture.repository.head_sha,
            "name": "Secret scan",
            "status": "completed",
            "conclusion": conclusion,
        }
        responses[f"repos/{self.fixture.repo_name}/actions/runs/108"] = {
            "id": 108,
            "run_attempt": 1,
            "event": "push",
            "path": ".github/workflows/secret-scan.yml",
            "head_sha": self.fixture.repository.head_sha,
            "head_repository": {"full_name": self.fixture.repo_name},
            "status": "completed",
            "conclusion": conclusion,
            "pull_requests": [],
        }

    def check_run(
        self, responses: dict[str, object], name: str
    ) -> dict[str, object]:
        endpoint = (
            f"repos/{self.fixture.repo_name}/commits/"
            f"{self.fixture.repository.head_sha}/check-runs?per_page=100&page=1"
        )
        return next(
            item
            for item in responses[endpoint]["check_runs"]
            if item["name"] == name and item["id"] != 3001
        )

    def test_later_push_secret_scan_run_does_not_replace_pull_request_evidence(
        self,
    ) -> None:
        responses = copy.deepcopy(self.fixture.responses)
        self.secret_scan_push_run(responses)

        result = self.collect(responses=responses)

        by_context = {
            entry["context"]: entry for entry in result["required_checks"]
        }
        self.assertEqual(
            by_context["Secret scan"]["check_run_id"],
            self.fixture.job_ids["Secret scan"],
        )
        self.assertEqual(result["workflow_runs"]["secret-scan.yml"]["id"], 107)

    def test_push_secret_scan_run_alone_is_not_pull_request_evidence(
        self,
    ) -> None:
        responses = copy.deepcopy(self.fixture.responses)
        self.secret_scan_push_run(responses)
        endpoint = (
            f"repos/{self.fixture.repo_name}/commits/"
            f"{self.fixture.repository.head_sha}/check-runs?per_page=100&page=1"
        )
        responses[endpoint]["check_runs"] = [
            item
            for item in responses[endpoint]["check_runs"]
            if item["id"] != self.fixture.job_ids["Secret scan"]
        ]

        with self.assertRaisesRegex(
            receipt.ReceiptNotReady,
            "Secret scan: no check run from its pull request workflow",
        ):
            self.collect(responses=responses)

    def test_failing_pull_request_secret_scan_is_not_masked_by_a_passing_push_run(
        self,
    ) -> None:
        responses = copy.deepcopy(self.fixture.responses)
        self.secret_scan_push_run(responses)
        failing = self.check_run(responses, "Secret scan")
        failing["conclusion"] = "failure"
        responses[
            f"repos/{self.fixture.repo_name}/actions/jobs/{failing['id']}"
        ]["conclusion"] = "failure"

        with self.assertRaisesRegex(
            receipt.ReceiptNotReady, "Secret scan: latest check is failure"
        ):
            self.collect(responses=responses)

    def test_same_named_run_from_another_workflow_stays_a_hard_error(
        self,
    ) -> None:
        responses = copy.deepcopy(self.fixture.responses)
        self.secret_scan_push_run(responses)
        responses[f"repos/{self.fixture.repo_name}/actions/runs/108"][
            "path"
        ] = ".github/workflows/other.yml"

        with self.assertRaisesRegex(
            receipt.ReceiptError, "Secret scan: run path was"
        ):
            self.collect(responses=responses)

    def test_rejects_unmapped_new_required_context(self) -> None:
        responses = copy.deepcopy(self.fixture.responses)
        branch = responses[
            f"repos/{self.fixture.repo_name}/branches/main"
        ]
        branch["protection"]["required_status_checks"]["checks"].append(
            {"context": "Unknown Required Gate", "app_id": self.fixture.app_id}
        )

        with self.assertRaisesRegex(receipt.ReceiptError, "Unknown Required Gate"):
            self.collect(responses=responses)

    def test_rejects_required_context_not_produced_by_expected_job(self) -> None:
        responses = copy.deepcopy(self.fixture.responses)
        endpoint = (
            f"repos/{self.fixture.repo_name}/commits/"
            f"{self.fixture.repository.head_sha}/check-runs?per_page=100&page=1"
        )
        check = next(
            item
            for item in responses[endpoint]["check_runs"]
            if item["name"] == "Docs"
        )
        check["id"] = 999999
        responses[
            f"repos/{self.fixture.repo_name}/actions/jobs/999999"
        ] = {
            "id": 999999,
            "run_id": self.fixture.run_ids["ci.yml"],
            "run_attempt": 1,
            "head_sha": self.fixture.repository.head_sha,
            "name": "Not Docs",
            "status": "completed",
            "conclusion": "success",
        }

        with self.assertRaisesRegex(receipt.ReceiptError, "expected workflow job"):
            self.collect(responses=responses)

    def test_rejects_disagreeing_ci_and_hull_candidates(self) -> None:
        artifacts = dict(self.fixture.artifacts)
        hull = dict(self.fixture.identities["conformance.yml"])
        hull["candidate_sha"] = self.fixture.repository.head_sha
        artifacts[502] = artifact(hull)

        with self.assertRaisesRegex(receipt.ReceiptError, "candidate identity"):
            self.collect(artifacts=artifacts)

    def test_rejects_a_forged_observed_base_parent(self) -> None:
        artifacts = dict(self.fixture.artifacts)
        ci = dict(self.fixture.identities["ci.yml"])
        ci["base_sha"] = self.fixture.repository.root_sha
        artifacts[501] = artifact(ci)

        with self.assertRaisesRegex(receipt.ReceiptError, "candidate identity"):
            self.collect(artifacts=artifacts)

    def test_rejects_forged_patch_identity_even_from_successful_workflow(self) -> None:
        artifacts = dict(self.fixture.artifacts)
        ci = dict(self.fixture.identities["ci.yml"])
        ci["patch_id"] = "f" * 40
        artifacts[501] = artifact(ci)

        with self.assertRaisesRegex(receipt.ReceiptError, "patch_id"):
            self.collect(artifacts=artifacts)

    def test_rejects_forged_exact_patch_digest(self) -> None:
        artifacts = dict(self.fixture.artifacts)
        ci = dict(self.fixture.identities["ci.yml"])
        ci["patch_digest"] = "f" * 64
        artifacts[501] = artifact(ci)

        with self.assertRaisesRegex(receipt.ReceiptError, "patch_digest"):
            self.collect(artifacts=artifacts)

    def test_rejects_trigger_from_an_unexpected_workflow(self) -> None:
        responses = copy.deepcopy(self.fixture.responses)
        trigger = responses[
            f"repos/{self.fixture.repo_name}/actions/runs/"
            f"{self.fixture.trigger_run_id}"
        ]
        trigger["path"] = ".github/workflows/untrusted.yml"

        with self.assertRaisesRegex(receipt.ReceiptError, "trigger workflow"):
            self.collect(responses=responses)

    def test_rejects_archive_with_extra_files(self) -> None:
        buffer = BytesIO()
        with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(
                receipt.IDENTITY_FILENAME,
                json.dumps(self.fixture.identities["ci.yml"]),
            )
            archive.writestr("extra.txt", "untrusted")
        artifacts = dict(self.fixture.artifacts)
        artifacts[501] = buffer.getvalue()

        with self.assertRaisesRegex(receipt.ReceiptError, "exactly one"):
            self.collect(artifacts=artifacts)


if __name__ == "__main__":
    unittest.main()
