"""Selection and manifest controls for the default-branch failure baseline."""
from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import unittest

from scripts import ci_failure_baseline as baseline


ARTIFACTS = ("junit-linux-full-1", "junit-linux-full-2")


def run_listing(rows: list[dict]) -> str:
    return json.dumps({"workflow_runs": rows})


def artifact_listing(names: list[str], *, expired: tuple[str, ...] = ()) -> str:
    return json.dumps(
        {
            "artifacts": [
                {"name": name, "expired": name in expired} for name in names
            ]
        }
    )


def run_row(identifier: str, sha: str, created: str) -> dict:
    return {
        "id": identifier,
        "html_url": f"https://example.invalid/{identifier}",
        "head_sha": sha,
        "head_branch": "main",
        "created_at": created,
    }


class FakeGh:
    """A recorded `gh` transcript keyed by the distinguishing argument."""

    def __init__(self, *, runs: str, artifacts: dict[str, str], root: Path) -> None:
        self.runs = runs
        self.artifacts = artifacts
        self.root = root
        self.downloaded: list[str] = []
        self.missing_documents: set[str] = set()

    def __call__(self, command):
        if command[1] == "api" and "/runs?" in command[-1]:
            return self.runs
        if command[1] == "api" and "/artifacts" in command[-1]:
            run_id = command[-1].split("/runs/")[1].split("/")[0]
            return self.artifacts[run_id]
        if command[1] == "run" and command[2] == "download":
            self.downloaded.append(command[3])
            destination = Path(command[command.index("--dir") + 1])
            for index, item in enumerate(command):
                if item != "--name":
                    continue
                name = command[index + 1]
                if name in self.missing_documents:
                    continue
                document = destination / name / "junit.xml"
                document.parent.mkdir(parents=True, exist_ok=True)
                document.write_text("<testsuites/>")
            return ""
        raise AssertionError(f"unexpected command: {command}")


class BaselineSelectionTests(unittest.TestCase):
    def prepare(self, gh: FakeGh, output: Path, **overrides):
        arguments = {
            "repository": "owner/name",
            "workflow": "heavy-e2e.yml",
            "branch": "main",
            "artifacts": ARTIFACTS,
            "search_runs": 5,
            "output": output,
            "runner": gh,
        }
        arguments.update(overrides)
        return baseline.prepare(**arguments)

    def test_the_newest_complete_run_becomes_the_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gh = FakeGh(
                runs=run_listing(
                    [
                        run_row("200", "b" * 40, "2026-09-19T03:31:04Z"),
                        run_row("100", "a" * 40, "2026-09-18T03:31:45Z"),
                    ]
                ),
                artifacts={
                    "200": artifact_listing(list(ARTIFACTS)),
                    "100": artifact_listing(list(ARTIFACTS)),
                },
                root=root,
            )
            manifest = self.prepare(gh, root / "out")
            written = json.loads((root / "out" / "baseline.json").read_text())
        self.assertEqual(manifest["run_id"], "200")
        self.assertEqual(manifest["head_sha"], "b" * 40)
        self.assertEqual(
            manifest["documents"],
            [f"{name}/junit.xml" for name in ARTIFACTS],
        )
        self.assertEqual(written, manifest)
        self.assertEqual(gh.downloaded, ["200"])

    def test_a_run_missing_a_shard_is_skipped_rather_than_used(self) -> None:
        """A partial baseline reports the missing shard's failures as new."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gh = FakeGh(
                runs=run_listing(
                    [
                        run_row("200", "b" * 40, "2026-09-19T03:31:04Z"),
                        run_row("100", "a" * 40, "2026-09-18T03:31:45Z"),
                    ]
                ),
                artifacts={
                    "200": artifact_listing([ARTIFACTS[0]]),
                    "100": artifact_listing(list(ARTIFACTS)),
                },
                root=root,
            )
            manifest = self.prepare(gh, root / "out")
        self.assertEqual(manifest["run_id"], "100")
        self.assertEqual(gh.downloaded, ["100"])

    def test_an_expired_artifact_does_not_qualify_a_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gh = FakeGh(
                runs=run_listing(
                    [run_row("200", "b" * 40, "2026-09-19T03:31:04Z")]
                ),
                artifacts={
                    "200": artifact_listing(
                        list(ARTIFACTS),
                        expired=(ARTIFACTS[1],),
                    )
                },
                root=root,
            )
            with self.assertRaisesRegex(ValueError, "retains every baseline"):
                self.prepare(gh, root / "out")

    def test_no_completed_run_fails_rather_than_reporting_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gh = FakeGh(runs=run_listing([]), artifacts={}, root=root)
            with self.assertRaisesRegex(ValueError, "no completed"):
                self.prepare(gh, root / "out")

    def test_a_run_on_another_branch_is_not_a_default_branch_baseline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = run_row("200", "b" * 40, "2026-09-19T03:31:04Z")
            row["head_branch"] = "agent/somewhere"
            gh = FakeGh(runs=run_listing([row]), artifacts={}, root=root)
            with self.assertRaisesRegex(ValueError, "no completed"):
                self.prepare(gh, root / "out")

    def test_a_downloaded_artifact_without_junit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gh = FakeGh(
                runs=run_listing(
                    [run_row("200", "b" * 40, "2026-09-19T03:31:04Z")]
                ),
                artifacts={"200": artifact_listing(list(ARTIFACTS))},
                root=root,
            )
            gh.missing_documents = {ARTIFACTS[1]}
            with self.assertRaisesRegex(ValueError, "has no junit.xml"):
                self.prepare(gh, root / "out")

    def test_a_red_baseline_run_still_qualifies(self) -> None:
        """The nightly is routinely red, and that redness is the evidence."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [run_row("200", "b" * 40, "2026-09-19T03:31:04Z")]
            rows[0]["conclusion"] = "failure"
            gh = FakeGh(
                runs=run_listing(rows),
                artifacts={"200": artifact_listing(list(ARTIFACTS))},
                root=root,
            )
            manifest = self.prepare(gh, root / "out")
        self.assertEqual(manifest["run_id"], "200")

    def test_paginated_artifact_pages_are_all_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paginated = (
                artifact_listing([ARTIFACTS[0]])
                + "\n"
                + artifact_listing([ARTIFACTS[1]])
            )
            gh = FakeGh(
                runs=run_listing(
                    [run_row("200", "b" * 40, "2026-09-19T03:31:04Z")]
                ),
                artifacts={"200": paginated},
                root=root,
            )
            manifest = self.prepare(gh, root / "out")
        self.assertEqual(manifest["run_id"], "200")


class BaselineArgumentTests(unittest.TestCase):
    def test_a_repository_is_required_outside_actions(self) -> None:
        with self.assertRaisesRegex(ValueError, "--repository is required"):
            baseline.main(["--repository", "", "--output", "unused"])

    def test_the_search_window_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "--search-runs must be positive"):
            baseline.main(
                [
                    "--repository",
                    "owner/name",
                    "--search-runs",
                    "0",
                    "--output",
                    "unused",
                ]
            )

    def test_the_default_artifacts_name_every_nightly_workspace_shard(
        self,
    ) -> None:
        """The producer and the consumer must not drift apart silently.

        A renamed or re-sharded nightly artifact would leave this script
        selecting no run, which fails loudly, but a *narrowed* default would
        quietly difference against a partial baseline instead.
        """
        self.assertEqual(baseline.DEFAULT_WORKFLOW, "heavy-e2e.yml")
        self.assertEqual(baseline.DEFAULT_BRANCH, "main")
        workflow = (
            Path(__file__).resolve().parent.parent
            / ".github"
            / "workflows"
            / baseline.DEFAULT_WORKFLOW
        ).read_text()
        shards = re.search(
            r"full-workspace:.*?matrix:\n\s+shard: \[(?P<shards>[^]]*)\]",
            workflow,
            re.DOTALL,
        )
        self.assertIsNotNone(shards, "heavy-e2e.yml lost its full-workspace matrix")
        values = [value.strip() for value in shards.group("shards").split(",")]
        self.assertIn(
            "name: junit-linux-full-${{ matrix.shard }}",
            workflow,
            "the nightly workspace JUnit artifact was renamed",
        )
        self.assertEqual(
            tuple(f"junit-linux-full-{value}" for value in values),
            baseline.DEFAULT_ARTIFACTS,
        )


if __name__ == "__main__":
    unittest.main()
