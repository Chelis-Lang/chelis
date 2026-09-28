#!/usr/bin/env python3
"""Tests for source and live validation of the [05-UNS-5] issue manifest."""

from __future__ import annotations

import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import validate_rejection_issue_manifest as validator
from capacity_census_liveness import IssueKind, IssueRecord, IssueState
from generate_rejection_registries import MANIFEST_REL, load_issue_manifest
from validate_rejection_issue_manifest import SourceManifestError, adjudicate, main

ROOT = Path(__file__).resolve().parent.parent


def issue_row(number: int) -> dict:
    return {"number": number, "kind": "issue", "state": "open"}


class ChangedRows(unittest.TestCase):
    """Real Git/manifest inputs; compiler source evidence is supplied separately."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.git("init", "--initial-branch=fixture")
        self.git("config", "user.name", "Rejection fixture")
        self.git("config", "user.email", "fixture@example.invalid")

    def git(self, *args: str, input: str | None = None) -> str:
        # Invocation-local: fixture cleanup must not race detached maintenance.
        return subprocess.run(
            ["git", "-c", "maintenance.auto=false", "-C", str(self.root), *args],
            input=input,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout.strip()

    def write_manifest(self, rows: list[dict]) -> None:
        path = self.root / MANIFEST_REL
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": 1, "issues": rows}) + "\n")

    def candidate(self, base: list[dict], current: list[dict]) -> str:
        self.write_manifest(base)
        self.git("add", ".")
        self.git("commit", "-m", "base")
        self.base = self.git("rev-parse", "HEAD")
        self.write_manifest(current)
        self.git("add", ".")
        self.git("commit", "--allow-empty", "-m", "candidate")
        head = self.git("rev-parse", "HEAD")
        tree = self.git("rev-parse", "HEAD^{tree}")
        merge = self.git(
            "commit-tree", tree, "-p", self.base, "-p", head,
            input="synthetic PR merge\n",
        )
        self.git("checkout", "--detach", merge)
        return head

    def source_evidence(self, root: Path) -> tuple[list[int], list[dict]]:
        self.assertEqual(root, self.root)
        path = root / MANIFEST_REL
        return load_issue_manifest(path), json.loads(path.read_text())["issues"]

    def run_validator(self, args: list[str], records: dict[int, IssueRecord]):
        output, errors = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(validator, "ROOT", self.root),
            mock.patch.object(
                validator, "validate_source_manifest", side_effect=self.source_evidence
            ) as source,
            mock.patch.object(
                validator, "fetch_issue", side_effect=records.get
            ) as fetch,
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            result = validator.main(args)
        return result, output.getvalue(), errors.getvalue(), source, fetch

    def test_pr_checks_added_row_and_does_not_fetch_unchanged_closed_row(self):
        head = self.candidate([issue_row(600)], [issue_row(600), issue_row(729)])
        result, output, _, source, fetch = self.run_validator(
            ["--pr-head", head],
            {
                600: IssueRecord(IssueKind.ISSUE, IssueState.CLOSED),
                729: IssueRecord(IssueKind.ISSUE, IssueState.OPEN),
            },
        )
        self.assertEqual(result, 0)
        self.assertIn(self.base, output)
        self.assertIn("729", output)
        self.assertTrue(output.rstrip().endswith("REJECTION ISSUE MANIFEST: PASS"))
        source.assert_called_once_with(self.root)
        fetch.assert_called_once_with(729)

    def test_each_invalid_added_authority_fails_closed(self):
        head = self.candidate([issue_row(600)], [issue_row(600), issue_row(729)])
        for record, reason in [
            (IssueRecord(IssueKind.ISSUE, IssueState.CLOSED), "STALE authority"),
            (IssueRecord(IssueKind.PULL_REQUEST, IssueState.OPEN), "WRONG OBJECT KIND"),
            (None, "UNRESOLVABLE authority"),
        ]:
            with self.subTest(reason=reason):
                result, output, errors, source, fetch = self.run_validator(
                    ["--pr-head", head], {729: record}
                )
                self.assertEqual(result, 1)
                self.assertIn(reason, errors)
                self.assertNotIn("REJECTION ISSUE MANIFEST: PASS", output)
                source.assert_called_once_with(self.root)
                fetch.assert_called_once_with(729)

    def test_renumbered_row_requires_the_successor_to_be_open(self):
        head = self.candidate([issue_row(600)], [issue_row(729)])
        result, _, errors, _, fetch = self.run_validator(
            ["--pr-head", head],
            {729: IssueRecord(IssueKind.ISSUE, IssueState.CLOSED)},
        )
        self.assertEqual(result, 1)
        self.assertIn("STALE authority", errors)
        fetch.assert_called_once_with(729)

    def test_removal_requires_source_agreement_but_no_live_lookup(self):
        head = self.candidate([issue_row(600), issue_row(729)], [issue_row(729)])
        result, output, _, source, fetch = self.run_validator(["--pr-head", head], {})
        self.assertEqual(result, 0)
        self.assertIn("checked 0", output)
        source.assert_called_once_with(self.root)
        fetch.assert_not_called()

    def test_unchanged_rows_do_not_trigger_live_lookup(self):
        head = self.candidate([issue_row(600)], [issue_row(600)])
        result, _, _, source, fetch = self.run_validator(["--pr-head", head], {})
        self.assertEqual(result, 0)
        source.assert_called_once_with(self.root)
        fetch.assert_not_called()

    def test_default_standing_mode_still_rejects_unchanged_closed_row(self):
        self.candidate([issue_row(600)], [issue_row(600)])
        result, _, errors, source, fetch = self.run_validator(
            [], {600: IssueRecord(IssueKind.ISSUE, IssueState.CLOSED)}
        )
        self.assertEqual(result, 1)
        self.assertIn("STALE authority", errors)
        source.assert_called_once_with(self.root)
        fetch.assert_called_once_with(600)

    def test_default_standing_mode_accepts_all_open_rows(self):
        self.candidate([issue_row(600)], [issue_row(600), issue_row(729)])
        result, output, _, source, fetch = self.run_validator(
            [], {n: IssueRecord(IssueKind.ISSUE, IssueState.OPEN) for n in [600, 729]}
        )
        self.assertEqual(result, 0)
        self.assertIn("checked 2", output)
        source.assert_called_once_with(self.root)
        self.assertEqual(fetch.call_args_list, [mock.call(600), mock.call(729)])

    def test_bad_pr_head_fails_before_source_or_network_validation(self):
        self.candidate([issue_row(600)], [issue_row(600), issue_row(729)])
        for head in ["0" * 40, "", "HEAD"]:
            with self.subTest(head=head):
                result, output, errors, source, fetch = self.run_validator(
                    ["--pr-head", head], {}
                )
                self.assertEqual(result, 1)
                self.assertIn("event pull-request head", errors)
                self.assertNotIn("REJECTION ISSUE MANIFEST: PASS", output)
                source.assert_not_called()
                fetch.assert_not_called()

    def test_nonmerge_candidate_is_refused(self):
        head = self.candidate([issue_row(600)], [issue_row(600), issue_row(729)])
        self.git("checkout", "--detach", head)
        result, _, errors, source, fetch = self.run_validator(["--pr-head", head], {})
        self.assertEqual(result, 1)
        self.assertIn("exactly two parents", errors)
        source.assert_not_called()
        fetch.assert_not_called()

    def test_malformed_base_manifest_is_not_treated_as_empty(self):
        head = self.candidate(
            [{"number": 600, "kind": "issue", "state": "closed"}], [issue_row(729)]
        )
        result, _, errors, source, fetch = self.run_validator(["--pr-head", head], {})
        self.assertEqual(result, 1)
        self.assertIn("state=open", errors)
        source.assert_not_called()
        fetch.assert_not_called()

    def test_invalid_candidate_row_is_rejected_even_with_no_new_number(self):
        head = self.candidate(
            [issue_row(600)], [{"number": 600, "kind": "issue", "state": "closed"}]
        )
        result, _, errors, source, fetch = self.run_validator(["--pr-head", head], {})
        self.assertEqual(result, 1)
        self.assertIn("state=open", errors)
        source.assert_called_once_with(self.root)
        fetch.assert_not_called()

    def test_missing_base_manifest_is_not_treated_as_empty(self):
        head = self.candidate([issue_row(600)], [issue_row(729)])
        tree = self.git("rev-parse", "HEAD^{tree}")
        self.git("checkout", "--detach", self.base)
        self.git("rm", str(MANIFEST_REL))
        self.git("commit", "-m", "base without manifest")
        missing_base = self.git("rev-parse", "HEAD")
        merge = self.git("commit-tree", tree, "-p", missing_base, "-p", head, input="merge\n")
        self.git("checkout", "--detach", merge)
        result, output, errors, source, fetch = self.run_validator(["--pr-head", head], {})
        self.assertEqual(result, 1)
        self.assertIn("REJECTION ISSUE MANIFEST: FAIL", errors)
        self.assertNotIn("REJECTION ISSUE MANIFEST: PASS", output)
        source.assert_not_called()
        fetch.assert_not_called()

    def test_missing_source_row_fails_before_empty_changed_row_success(self):
        head = self.candidate([issue_row(600)], [issue_row(600)])
        with (
            mock.patch.object(validator, "ROOT", self.root),
            mock.patch.object(
                validator, "validate_source_manifest",
                side_effect=SourceManifestError(["issue manifest is missing source-cited rows: [729]"]),
            ),
            mock.patch.object(validator, "fetch_issue") as fetch,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()) as errors,
        ):
            self.assertEqual(validator.main(["--pr-head", head]), 1)
        self.assertIn("missing source-cited rows", errors.getvalue())
        fetch.assert_not_called()


class Adjudicate(unittest.TestCase):
    def test_open_issues_pass(self) -> None:
        rows = [{"number": 705}, {"number": 879}]
        records = {
            705: IssueRecord(IssueKind.ISSUE, IssueState.OPEN),
            879: IssueRecord(IssueKind.ISSUE, IssueState.OPEN),
        }
        self.assertEqual(adjudicate(rows, records), [])

    def test_closed_issue_fails(self) -> None:
        problems = adjudicate(
            [{"number": 944}],
            {944: IssueRecord(IssueKind.ISSUE, IssueState.CLOSED)},
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("CLOSED", problems[0])

    def test_pull_request_fails_even_when_open(self) -> None:
        problems = adjudicate(
            [{"number": 1}],
            {1: IssueRecord(IssueKind.PULL_REQUEST, IssueState.OPEN)},
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("PULL REQUEST", problems[0])

    def test_missing_number_fails_closed(self) -> None:
        problems = adjudicate([{"number": 999_999_999}], {})
        self.assertEqual(len(problems), 1)
        self.assertIn("UNRESOLVABLE", problems[0])

    @mock.patch("validate_rejection_issue_manifest.fetch_issue")
    @mock.patch("validate_rejection_issue_manifest.validate_source_manifest")
    def test_main_fetches_every_standing_row_without_rederiving_source_evidence(
        self,
        validate_source_manifest: mock.Mock,
        fetch_issue: mock.Mock,
    ) -> None:
        expected = load_issue_manifest(ROOT / MANIFEST_REL)
        validate_source_manifest.return_value = (
            expected,
            [
                {"number": number, "kind": "issue", "state": "open"}
                for number in expected
            ],
        )
        fetch_issue.return_value = IssueRecord(IssueKind.ISSUE, IssueState.OPEN)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main([]), 0)
        validate_source_manifest.assert_called_once_with(ROOT)
        self.assertEqual(
            [call.args[0] for call in fetch_issue.call_args_list],
            expected,
        )

    @mock.patch("validate_rejection_issue_manifest.fetch_issue")
    @mock.patch("validate_rejection_issue_manifest.validate_source_manifest")
    def test_main_fails_before_liveness_when_fresh_derivation_disagrees(
        self,
        validate_source_manifest: mock.Mock,
        fetch_issue: mock.Mock,
    ) -> None:
        validate_source_manifest.side_effect = SourceManifestError(
            ["issue manifest contains uncited stale rows: [879]"]
        )
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(main([]), 1)
        self.assertIn("uncited stale rows", stderr.getvalue())
        self.assertIn("REJECTION ISSUE MANIFEST: FAIL", stderr.getvalue())
        fetch_issue.assert_not_called()


if __name__ == "__main__":
    unittest.main()
