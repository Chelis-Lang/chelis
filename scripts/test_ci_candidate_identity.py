from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts import ci_candidate_identity as identity


def git(repository: Path, *arguments: str, input_text: str | None = None) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


class CandidateRepository:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)
        git(self.path, "init", "-b", "main")
        git(self.path, "config", "user.name", "CI Test")
        git(self.path, "config", "user.email", "ci@example.invalid")
        (self.path / "shared.txt").write_text("root\n")
        git(self.path, "add", "shared.txt")
        git(self.path, "commit", "-m", "root")
        self.root_sha = git(self.path, "rev-parse", "HEAD")
        git(self.path, "switch", "-c", "feature")
        (self.path / "feature.txt").write_text("candidate\n")
        git(self.path, "add", "feature.txt")
        git(self.path, "commit", "-m", "feature")
        self.head_sha = git(self.path, "rev-parse", "HEAD")
        git(self.path, "switch", "main")
        (self.path / "base.txt").write_text("base\n")
        git(self.path, "add", "base.txt")
        git(self.path, "commit", "-m", "base")
        self.base_sha = git(self.path, "rev-parse", "HEAD")
        git(self.path, "merge", "--no-ff", "feature", "-m", "synthetic candidate")
        self.candidate_sha = git(self.path, "rev-parse", "HEAD")

    def close(self) -> None:
        self.temporary.cleanup()


class CandidateIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = CandidateRepository()

    def tearDown(self) -> None:
        self.repository.close()

    def build(self) -> dict[str, object]:
        return identity.build_identity(
            repository_path=self.repository.path,
            repository="Chelis-Lang/chelis",
            workflow_file="ci.yml",
            run_id=101,
            run_attempt=2,
            pr_number=2093,
            head_sha=self.repository.head_sha,
            base_ref="main",
            base_sha=self.repository.base_sha,
            candidate_sha=self.repository.candidate_sha,
        )

    def test_records_exact_candidate_and_rebase_comparison_material(self) -> None:
        payload = self.build()

        self.assertEqual(payload["schema"], identity.SCHEMA)
        self.assertEqual(
            payload["candidate_parents"],
            [self.repository.base_sha, self.repository.head_sha],
        )
        self.assertEqual(
            payload["patch_base_sha"],
            git(
                self.repository.path,
                "merge-base",
                self.repository.base_sha,
                self.repository.head_sha,
            ),
        )
        self.assertEqual(payload["changed_paths"], ["feature.txt"])
        self.assertRegex(str(payload["changed_paths_sha256"]), r"^[0-9a-f]{64}$")
        self.assertRegex(str(payload["patch_id"]), r"^[0-9a-f]{40}$")
        self.assertRegex(str(payload["patch_digest"]), r"^[0-9a-f]{64}$")
        self.assertEqual(payload["workflow_run_id"], 101)
        self.assertEqual(payload["workflow_run_attempt"], 2)

    def test_candidate_first_parent_survives_a_stale_event_base(self) -> None:
        payload = identity.build_identity(
            repository_path=self.repository.path,
            repository="Chelis-Lang/chelis",
            workflow_file="ci.yml",
            run_id=101,
            run_attempt=1,
            pr_number=2099,
            head_sha=self.repository.head_sha,
            base_ref="main",
            base_sha=None,
            candidate_sha=self.repository.candidate_sha,
        )

        self.assertNotEqual(self.repository.root_sha, self.repository.base_sha)
        self.assertEqual(payload["base_sha"], self.repository.base_sha)
        self.assertEqual(
            payload["candidate_parents"],
            [self.repository.base_sha, self.repository.head_sha],
        )

    def test_explicit_expected_base_still_rejects_a_mismatch(self) -> None:
        with self.assertRaisesRegex(
            identity.IdentityError,
            "candidate base parent does not match the expected base",
        ):
            identity.build_identity(
                repository_path=self.repository.path,
                repository="Chelis-Lang/chelis",
                workflow_file="ci.yml",
                run_id=101,
                run_attempt=1,
                pr_number=2099,
                head_sha=self.repository.head_sha,
                base_ref="main",
                base_sha=self.repository.root_sha,
                candidate_sha=self.repository.candidate_sha,
            )

    def test_exact_patch_digest_survives_an_unrelated_clean_rebase(self) -> None:
        original = self.build()
        git(
            self.repository.path,
            "switch",
            "-c",
            "rebased-feature",
            self.repository.head_sha,
        )
        git(
            self.repository.path,
            "rebase",
            "--onto",
            self.repository.base_sha,
            self.repository.root_sha,
        )
        rebased_head = git(self.repository.path, "rev-parse", "HEAD")
        git(
            self.repository.path,
            "switch",
            "-c",
            "rebased-candidate",
            self.repository.base_sha,
        )
        git(
            self.repository.path,
            "merge",
            "--no-ff",
            "rebased-feature",
            "-m",
            "rebased synthetic candidate",
        )
        rebased_candidate = git(self.repository.path, "rev-parse", "HEAD")

        rebased = identity.build_identity(
            repository_path=self.repository.path,
            repository="Chelis-Lang/chelis",
            workflow_file="ci.yml",
            run_id=102,
            run_attempt=1,
            pr_number=2093,
            head_sha=rebased_head,
            base_ref="main",
            base_sha=self.repository.base_sha,
            candidate_sha=rebased_candidate,
        )

        self.assertEqual(rebased["patch_id"], original["patch_id"])
        self.assertEqual(rebased["patch_digest"], original["patch_digest"])
        self.assertEqual(rebased["changed_paths"], original["changed_paths"])

    def test_exact_patch_digest_detects_whitespace_only_patch_change(self) -> None:
        original = self.build()
        git(
            self.repository.path,
            "switch",
            "-c",
            "whitespace-feature",
            self.repository.head_sha,
        )
        (self.repository.path / "feature.txt").write_text("candidate \n")
        git(self.repository.path, "add", "feature.txt")
        git(self.repository.path, "commit", "-m", "change whitespace")
        whitespace_head = git(self.repository.path, "rev-parse", "HEAD")
        git(
            self.repository.path,
            "switch",
            "-c",
            "whitespace-candidate",
            self.repository.base_sha,
        )
        git(
            self.repository.path,
            "merge",
            "--no-ff",
            "whitespace-feature",
            "-m",
            "whitespace synthetic candidate",
        )
        whitespace_candidate = git(self.repository.path, "rev-parse", "HEAD")

        changed = identity.build_identity(
            repository_path=self.repository.path,
            repository="Chelis-Lang/chelis",
            workflow_file="ci.yml",
            run_id=103,
            run_attempt=1,
            pr_number=2093,
            head_sha=whitespace_head,
            base_ref="main",
            base_sha=self.repository.base_sha,
            candidate_sha=whitespace_candidate,
        )

        self.assertNotEqual(changed["patch_digest"], original["patch_digest"])

    def test_rejects_candidate_whose_second_parent_is_not_the_pr_head(self) -> None:
        with self.assertRaisesRegex(
            identity.IdentityError,
            "candidate head parent does not match the pull request head",
        ):
            identity.build_identity(
                repository_path=self.repository.path,
                repository="Chelis-Lang/chelis",
                workflow_file="ci.yml",
                run_id=101,
                run_attempt=1,
                pr_number=2093,
                head_sha=self.repository.base_sha,
                base_ref="main",
                base_sha=self.repository.base_sha,
                candidate_sha=self.repository.candidate_sha,
            )

    def test_reads_raw_parent_headers_at_a_shallow_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as source_directory:
            with tempfile.TemporaryDirectory() as clone_directory:
                source = Path(source_directory)
                shallow = Path(clone_directory)
                git(source, "init", "-b", "main")
                git(source, "config", "user.name", "CI Test")
                git(source, "config", "user.email", "ci@example.invalid")
                (source / "shared.txt").write_text("root\n")
                git(source, "add", "shared.txt")
                git(source, "commit", "-m", "root")
                (source / "base.txt").write_text("base\n")
                git(source, "add", "base.txt")
                git(source, "commit", "-m", "base")
                base_sha = git(source, "rev-parse", "HEAD")
                git(source, "switch", "-c", "feature")
                (source / "feature.txt").write_text("candidate\n")
                git(source, "add", "feature.txt")
                git(source, "commit", "-m", "feature")
                head_sha = git(source, "rev-parse", "HEAD")
                git(source, "branch", "event-base", base_sha)
                git(source, "branch", "event-head", head_sha)
                git(source, "switch", "-c", "candidate", base_sha)
                git(
                    source,
                    "merge",
                    "--no-ff",
                    "feature",
                    "-m",
                    "synthetic candidate",
                )
                candidate_sha = git(source, "rev-parse", "HEAD")

                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--no-local",
                        "--depth=1",
                        "--branch",
                        "candidate",
                        source.as_uri(),
                        str(shallow),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                git(
                    shallow,
                    "fetch",
                    "--depth=1",
                    "origin",
                    "refs/heads/event-base",
                )
                git(
                    shallow,
                    "fetch",
                    "--depth=2",
                    "origin",
                    "refs/heads/event-head",
                )
                self.assertEqual(
                    git(shallow, "show", "-s", "--format=%P", candidate_sha),
                    "",
                )

                payload = identity.build_identity(
                    repository_path=shallow,
                    repository="Chelis-Lang/chelis",
                    workflow_file="ci.yml",
                    run_id=101,
                    run_attempt=1,
                    pr_number=2098,
                    head_sha=head_sha,
                    base_ref="main",
                    base_sha=None,
                    candidate_sha=candidate_sha,
                )

                self.assertEqual(
                    payload["candidate_parents"], [base_sha, head_sha]
                )

    def test_rejects_non_pr_workflow_file_and_malformed_inputs(self) -> None:
        cases = (
            {"workflow_file": "../ci.yml"},
            {"repository": "chelis"},
            {"base_ref": "../main"},
            {"run_id": 0},
            {"run_attempt": 0},
            {"pr_number": 0},
            {"base_sha": "not-a-sha"},
        )
        defaults: dict[str, object] = {
            "repository_path": self.repository.path,
            "repository": "Chelis-Lang/chelis",
            "workflow_file": "ci.yml",
            "run_id": 101,
            "run_attempt": 1,
            "pr_number": 2093,
            "head_sha": self.repository.head_sha,
            "base_ref": "main",
            "base_sha": self.repository.base_sha,
            "candidate_sha": self.repository.candidate_sha,
        }
        for change in cases:
            with self.subTest(change=change):
                arguments = defaults | change
                with self.assertRaises(identity.IdentityError):
                    identity.build_identity(**arguments)

    def test_write_identity_uses_canonical_json(self) -> None:
        output = self.repository.path / "identity.json"
        payload = self.build()

        identity.write_identity(payload, output)

        self.assertEqual(json.loads(output.read_text()), payload)
        self.assertTrue(output.read_text().endswith("\n"))
        self.assertEqual(
            output.read_text(),
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )


if __name__ == "__main__":
    unittest.main()
