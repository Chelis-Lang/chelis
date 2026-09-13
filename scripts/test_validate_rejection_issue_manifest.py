#!/usr/bin/env python3
"""Tests for source and live validation of the [05-UNS-5] issue manifest."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from capacity_census_liveness import IssueKind, IssueRecord, IssueState
from generate_rejection_registries import MANIFEST_REL, load_issue_manifest
from validate_rejection_issue_manifest import SourceManifestError, adjudicate, main

ROOT = Path(__file__).resolve().parent.parent


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
            self.assertEqual(main(), 0)
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
            self.assertEqual(main(), 1)
        self.assertIn("uncited stale rows", stderr.getvalue())
        self.assertIn("REJECTION ISSUE MANIFEST: FAIL", stderr.getvalue())
        fetch_issue.assert_not_called()


if __name__ == "__main__":
    unittest.main()
