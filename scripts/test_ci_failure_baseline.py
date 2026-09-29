"""Selection and manifest controls for the default-branch failure baseline."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from scripts import ci_failure_baseline as baseline


ARTIFACTS = baseline.DEFAULT_ARTIFACTS
SCOPE_ARTIFACT = "linux-extended-dispatch-scope"


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


def run_row(
    identifier: str, sha: str, created: str, *, event: str = "schedule",
) -> dict:
    return {
        "id": identifier,
        "html_url": f"https://example.invalid/{identifier}",
        "head_sha": sha,
        "head_branch": "main",
        "created_at": created,
        "event": event,
    }


def scope_receipt(
    identifier: str, sha: str, *, scope: str = "all",
) -> str:
    return json.dumps(
        {
            "version": 1,
            "event": "workflow_dispatch",
            "scope": scope,
            "run_id": identifier,
            "head_sha": sha,
        }
    )


class FakeGh:
    """A recorded `gh` transcript keyed by the distinguishing argument."""

    def __init__(self, *, runs: str, artifacts: dict[str, str], root: Path) -> None:
        self.runs = runs
        self.artifacts = artifacts
        self.root = root
        self.downloaded: list[str] = []
        self.download_names: list[tuple[str, tuple[str, ...]]] = []
        self.missing_documents: set[str] = set()
        self.scope_receipts: dict[str, str] = {}

    def __call__(self, command):
        if command[1] == "api" and "/runs?" in command[-1]:
            return self.runs
        if command[1] == "api" and "/artifacts" in command[-1]:
            run_id = command[-1].split("/runs/")[1].split("/")[0]
            return self.artifacts[run_id]
        if command[1] == "run" and command[2] == "download":
            run_id = command[3]
            self.downloaded.append(run_id)
            names = tuple(
                command[index + 1]
                for index, item in enumerate(command)
                if item == "--name"
            )
            self.download_names.append((run_id, names))
            destination = Path(command[command.index("--dir") + 1])
            for index, item in enumerate(command):
                if item != "--name":
                    continue
                name = command[index + 1]
                if name in self.missing_documents:
                    continue
                document = (
                    destination / name if len(names) > 1 else destination
                ) / ("scope.json" if name == SCOPE_ARTIFACT else "junit.xml")
                document.parent.mkdir(parents=True, exist_ok=True)
                document.write_text(
                    self.scope_receipts.get(run_id, "")
                    if name == SCOPE_ARTIFACT
                    else "<testsuites/>"
                )
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
                artifact_listing(list(ARTIFACTS[:2]))
                + "\n"
                + artifact_listing(list(ARTIFACTS[2:]))
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

    def test_a_scoped_dispatch_with_all_four_junits_is_not_a_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha = "b" * 40
            gh = FakeGh(
                runs=run_listing([
                    run_row("200", sha, "2026-09-19T03:31:04Z",
                            event="workflow_dispatch"),
                    run_row("100", "a" * 40, "2026-09-18T03:31:45Z"),
                ]),
                artifacts={
                    "200": artifact_listing(
                        [*baseline.DEFAULT_ARTIFACTS, SCOPE_ARTIFACT]
                    ),
                    "100": artifact_listing(list(baseline.DEFAULT_ARTIFACTS)),
                },
                root=root,
            )
            gh.scope_receipts["200"] = scope_receipt(
                "200", sha, scope="runtime-representation"
            )
            manifest = self.prepare(
                gh, root / "out", artifacts=baseline.DEFAULT_ARTIFACTS
            )
        self.assertEqual(manifest["run_id"], "100")
        self.assertEqual(gh.download_names[0], ("200", (SCOPE_ARTIFACT,)))
        self.assertNotIn(("200", baseline.DEFAULT_ARTIFACTS), gh.download_names)

    def test_a_full_dispatch_with_exact_receipt_and_four_junits_qualifies(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha = "b" * 40
            gh = FakeGh(
                runs=run_listing([
                    run_row("200", sha, "2026-09-19T03:31:04Z",
                            event="workflow_dispatch")
                ]),
                artifacts={
                    "200": artifact_listing(
                        [*baseline.DEFAULT_ARTIFACTS, SCOPE_ARTIFACT]
                    )
                },
                root=root,
            )
            gh.scope_receipts["200"] = scope_receipt("200", sha)
            manifest = self.prepare(
                gh, root / "out", artifacts=baseline.DEFAULT_ARTIFACTS
            )
        self.assertEqual(manifest["run_id"], "200")
        self.assertEqual(gh.downloaded, ["200", "200"])

    def test_a_legacy_schedule_needs_four_junits_but_no_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gh = FakeGh(
                runs=run_listing([
                    run_row("100", "a" * 40, "2026-09-18T03:31:45Z")
                ]),
                artifacts={
                    "100": artifact_listing(list(baseline.DEFAULT_ARTIFACTS))
                },
                root=root,
            )
            manifest = self.prepare(
                gh, root / "out", artifacts=baseline.DEFAULT_ARTIFACTS
            )
        self.assertEqual(manifest["run_id"], "100")
        self.assertEqual(gh.downloaded, ["100"])

    def test_dispatch_receipt_must_bind_the_run_and_head_and_full_scope(
        self,
    ) -> None:
        sha = "b" * 40
        for receipt in (
            scope_receipt("201", sha),
            scope_receipt("200", "c" * 40),
            scope_receipt("200", sha, scope="runtime-representation"),
            json.dumps({"version": 1, "event": "schedule",
                        "scope": "all", "run_id": "200", "head_sha": sha}),
            json.dumps({"version": 2, "event": "workflow_dispatch",
                        "scope": "all", "run_id": "200", "head_sha": sha}),
            json.dumps({"version": True, "event": "workflow_dispatch",
                        "scope": "all", "run_id": "200", "head_sha": sha}),
            "{",
            "",
        ):
            with self.subTest(receipt=receipt), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                gh = FakeGh(
                    runs=run_listing([
                        run_row("200", sha, "2026-09-19T03:31:04Z",
                                event="workflow_dispatch")
                    ]),
                    artifacts={
                        "200": artifact_listing(
                            [*baseline.DEFAULT_ARTIFACTS, SCOPE_ARTIFACT]
                        )
                    },
                    root=root,
                )
                gh.scope_receipts["200"] = receipt
                with self.assertRaisesRegex(ValueError, "retains every baseline"):
                    self.prepare(
                        gh, root / "out", artifacts=baseline.DEFAULT_ARTIFACTS
                    )
                self.assertFalse((root / "out" / "baseline.json").exists())

    def test_dispatch_missing_or_ambiguous_receipt_is_rejected(self) -> None:
        for receipt_artifacts in (
            list(baseline.DEFAULT_ARTIFACTS),
            [*baseline.DEFAULT_ARTIFACTS, SCOPE_ARTIFACT, SCOPE_ARTIFACT],
        ):
            with self.subTest(artifacts=receipt_artifacts), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                gh = FakeGh(
                    runs=run_listing([
                        run_row("200", "b" * 40, "2026-09-19T03:31:04Z",
                                event="workflow_dispatch")
                    ]),
                    artifacts={"200": artifact_listing(receipt_artifacts)},
                    root=root,
                )
                gh.scope_receipts["200"] = scope_receipt("200", "b" * 40)
                with self.assertRaisesRegex(ValueError, "retains every baseline"):
                    self.prepare(
                        gh, root / "out", artifacts=baseline.DEFAULT_ARTIFACTS
                    )

    def test_listed_receipt_without_scope_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gh = FakeGh(
                runs=run_listing([
                    run_row("200", "b" * 40, "2026-09-19T03:31:04Z",
                            event="workflow_dispatch")
                ]),
                artifacts={
                    "200": artifact_listing(
                        [*baseline.DEFAULT_ARTIFACTS, SCOPE_ARTIFACT]
                    )
                },
                root=root,
            )
            gh.missing_documents.add(SCOPE_ARTIFACT)
            with self.assertRaisesRegex(ValueError, "retains every baseline"):
                self.prepare(gh, root / "out", artifacts=baseline.DEFAULT_ARTIFACTS)

    def test_default_workflow_cannot_narrow_the_required_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gh = FakeGh(runs=run_listing([]), artifacts={}, root=root)
            with self.assertRaisesRegex(ValueError, "all four JUnit"):
                self.prepare(gh, root / "out", artifacts=ARTIFACTS[:2])

    def test_unknown_event_is_not_treated_as_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gh = FakeGh(
                runs=run_listing([
                    run_row("200", "b" * 40, "2026-09-19T03:31:04Z",
                            event="push")
                ]),
                artifacts={
                    "200": artifact_listing(list(baseline.DEFAULT_ARTIFACTS))
                },
                root=root,
            )
            with self.assertRaisesRegex(ValueError, "no completed"):
                self.prepare(gh, root / "out", artifacts=baseline.DEFAULT_ARTIFACTS)


class BaselineBaseContainmentTests(unittest.TestCase):
    """The consumer refuses a baseline the base does not contain."""

    def runs(self) -> str:
        return run_listing(
            [
                run_row("300", "c" * 40, "2026-09-19T03:31:04Z"),
                run_row("200", "b" * 40, "2026-09-18T03:31:45Z"),
                run_row("100", "a" * 40, "2026-09-17T03:32:46Z"),
            ]
        )

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

    def test_a_run_the_base_does_not_contain_is_passed_over(self) -> None:
        """A merge ref is not recomputed, so the newest run is often ahead."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gh = FakeGh(
                runs=self.runs(),
                artifacts={
                    key: artifact_listing(list(ARTIFACTS))
                    for key in ("100", "200", "300")
                },
                root=root,
            )
            manifest = self.prepare(
                gh,
                root / "out",
                base_sha="z" * 40,
                repo=root,
                contains=lambda repo, commit, base: commit != "c" * 40,
            )
        self.assertEqual(manifest["run_id"], "200")
        self.assertEqual(gh.downloaded, ["200"])

    def test_no_contained_run_names_the_base_in_its_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gh = FakeGh(
                runs=self.runs(),
                artifacts={
                    key: artifact_listing(list(ARTIFACTS))
                    for key in ("100", "200", "300")
                },
                root=root,
            )
            with self.assertRaisesRegex(ValueError, "not contained in the candidate base"):
                self.prepare(
                    gh,
                    root / "out",
                    base_sha="z" * 40,
                    repo=root,
                    contains=lambda repo, commit, base: False,
                )

    def test_without_a_base_the_newest_run_is_still_taken(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gh = FakeGh(
                runs=self.runs(),
                artifacts={
                    key: artifact_listing(list(ARTIFACTS))
                    for key in ("100", "200", "300")
                },
                root=root,
            )
            manifest = self.prepare(gh, root / "out")
        self.assertEqual(manifest["run_id"], "300")

    def test_base_containment_is_measured_on_real_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for command in (
                ["git", "init", "--quiet", "--initial-branch=main", str(root)],
                ["git", "-C", str(root), "config", "user.email", "t@example.invalid"],
                ["git", "-C", str(root), "config", "user.name", "T"],
            ):
                subprocess.run(command, check=True, capture_output=True)

            def commit(message: str) -> str:
                (root / message).write_text(message)
                subprocess.run(
                    ["git", "-C", str(root), "add", "-A"],
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    ["git", "-C", str(root), "commit", "--quiet", "-m", message],
                    check=True,
                    capture_output=True,
                )
                return subprocess.run(
                    ["git", "-C", str(root), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()

            first = commit("first")
            second = commit("second")
            self.assertTrue(baseline.base_contains(root, first, second))
            self.assertTrue(baseline.base_contains(root, second, second))
            self.assertFalse(baseline.base_contains(root, second, first))
            self.assertFalse(baseline.base_contains(root, "e" * 40, second))

    def test_an_unresolvable_base_is_loud_and_an_unresolvable_run_is_not(
        self,
    ) -> None:
        """The two commits fail differently on purpose.

        An absent baseline commit is genuinely not contained, so the next run
        is tried. An absent base disqualifies every run alike, and the caller
        would otherwise blame artifact retention for something that is not
        about retention at all.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for command in (
                ["git", "init", "--quiet", "--initial-branch=main", str(root)],
                [
                    "git", "-C", str(root), "config",
                    "user.email", "t@example.invalid",
                ],
                ["git", "-C", str(root), "config", "user.name", "T"],
            ):
                subprocess.run(command, check=True, capture_output=True)
            (root / "f").write_text("f")
            subprocess.run(
                ["git", "-C", str(root), "add", "-A"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "commit", "--quiet", "-m", "f"],
                check=True,
                capture_output=True,
            )
            head = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertFalse(baseline.base_contains(root, "e" * 40, head))
            with self.assertRaisesRegex(
                ValueError, "candidate base .* not present"
            ):
                baseline.base_contains(root, head, "e" * 40)

    def test_the_plan_supplies_the_base_and_a_planless_one_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = Path(tmp) / "plan.json"
            plan.write_text(json.dumps({"base_sha": "a" * 40}))
            self.assertEqual(baseline.plan_base_sha(plan), "a" * 40)
            plan.write_text(json.dumps({}))
            with self.assertRaisesRegex(ValueError, "no base_sha"):
                baseline.plan_base_sha(plan)


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
