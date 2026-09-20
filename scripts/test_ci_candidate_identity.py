from __future__ import annotations

import contextlib
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


class AdvancedTargetRepository:
    """A pull request whose target advanced past its branch point.

    This is chelis#2228's topology. The candidate's first parent is the
    target snapshot GitHub merged, which sits one commit ahead of the
    branch point, so resolving the patch base requires walking the target
    back one step. `clone(connected=False)` reproduces the backstop fetch
    that grafts the target and hides that step; `clone(connected=True)`
    reproduces the repaired depth-less fetch.
    """

    #: `github.event.pull_request.commits`, the head deepening's budget.
    COMMITS = 2

    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)
        git(self.path, "init", "-b", "main")
        git(self.path, "config", "user.name", "CI Test")
        git(self.path, "config", "user.email", "ci@example.invalid")
        (self.path / "shared.txt").write_text("root\n")
        git(self.path, "add", "shared.txt")
        git(self.path, "commit", "-m", "root")
        (self.path / "base.txt").write_text("base\n")
        git(self.path, "add", "base.txt")
        git(self.path, "commit", "-m", "branch point")
        self.branch_point_sha = git(self.path, "rev-parse", "HEAD")
        git(self.path, "switch", "-c", "event-head")
        for revision in range(self.COMMITS):
            (self.path / "feature.txt").write_text(f"candidate {revision}\n")
            git(self.path, "add", "feature.txt")
            git(self.path, "commit", "-m", f"feature {revision}")
        self.head_sha = git(self.path, "rev-parse", "HEAD")
        git(self.path, "switch", "main")
        (self.path / "target.txt").write_text("target advanced\n")
        git(self.path, "add", "target.txt")
        git(self.path, "commit", "-m", "target advances past the branch point")
        self.target_sha = git(self.path, "rev-parse", "HEAD")
        git(self.path, "switch", "-c", "candidate")
        git(
            self.path,
            "merge",
            "--no-ff",
            "event-head",
            "-m",
            "synthetic candidate",
        )
        self.candidate_sha = git(self.path, "rev-parse", "HEAD")

    def __enter__(self) -> "AdvancedTargetRepository":
        return self

    def __exit__(self, *exception: object) -> None:
        self.temporary.cleanup()

    @contextlib.contextmanager
    def clone(self, *, connected: bool):
        """Replay the `changes` job's fetches into a depth-one checkout."""

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--no-local",
                    "--depth=1",
                    "--branch",
                    "candidate",
                    self.path.as_uri(),
                    str(destination),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            git(
                destination,
                "fetch",
                f"--depth={self.COMMITS + 1}",
                "origin",
                "refs/heads/event-head",
            )
            backstop = ["fetch", "--no-tags"]
            if not connected:
                backstop.append("--depth=1")
            git(destination, *backstop, "origin", "refs/heads/main")
            yield destination

    def build(self, repository_path: Path) -> dict[str, object]:
        return identity.build_identity(
            repository_path=repository_path,
            repository="Chelis-Lang/chelis",
            workflow_file="ci.yml",
            run_id=105,
            run_attempt=1,
            pr_number=2228,
            head_sha=self.head_sha,
            base_ref="main",
            base_sha=None,
            candidate_sha=self.candidate_sha,
        )


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

    def test_a_grafted_target_snapshot_is_reported_as_truncation(self) -> None:
        with AdvancedTargetRepository() as source:
            with source.clone(connected=False) as shallow:
                with self.assertRaises(identity.IdentityError) as raised:
                    source.build(shallow)

        message = str(raised.exception)
        self.assertIn("the patch base is not resolvable", message)
        self.assertIn("this checkout is shallow", message)
        self.assertIn("is a shallow boundary whose recorded parent", message)
        self.assertNotIn("share no history", message)

    def test_a_connected_target_snapshot_resolves_the_branch_point(
        self,
    ) -> None:
        with AdvancedTargetRepository() as source:
            with source.clone(connected=True) as repaired:
                payload = source.build(repaired)

                self.assertEqual(
                    git(repaired, "rev-parse", "--is-shallow-repository"),
                    "true",
                )

        self.assertEqual(payload["patch_base_sha"], source.branch_point_sha)
        self.assertEqual(payload["base_sha"], source.target_sha)
        self.assertEqual(payload["changed_paths"], ["feature.txt"])

    def test_unrelated_parents_still_fail_in_a_complete_clone(self) -> None:
        """Negative control: parents with no shared history must still fail."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            git(path, "init", "-b", "main")
            git(path, "config", "user.name", "CI Test")
            git(path, "config", "user.email", "ci@example.invalid")
            (path / "base.txt").write_text("base\n")
            git(path, "add", "base.txt")
            git(path, "commit", "-m", "base")
            base_sha = git(path, "rev-parse", "HEAD")
            git(path, "checkout", "--orphan", "unrelated")
            git(path, "rm", "-rf", ".")
            (path / "feature.txt").write_text("candidate\n")
            git(path, "add", "feature.txt")
            git(path, "commit", "-m", "unrelated feature")
            head_sha = git(path, "rev-parse", "HEAD")
            git(path, "switch", "main")
            git(
                path,
                "merge",
                "--no-ff",
                "--allow-unrelated-histories",
                "unrelated",
                "-m",
                "synthetic candidate",
            )
            candidate_sha = git(path, "rev-parse", "HEAD")
            self.assertEqual(
                git(path, "rev-parse", "--is-shallow-repository"), "false"
            )

            with self.assertRaises(identity.IdentityError) as raised:
                identity.build_identity(
                    repository_path=path,
                    repository="Chelis-Lang/chelis",
                    workflow_file="ci.yml",
                    run_id=104,
                    run_attempt=1,
                    pr_number=2228,
                    head_sha=head_sha,
                    base_ref="main",
                    base_sha=base_sha,
                    candidate_sha=candidate_sha,
                )

        message = str(raised.exception)
        self.assertIn("the candidate's parents share no history", message)
        self.assertIn("in a complete clone", message)
        self.assertNotIn("shallow", message)

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
