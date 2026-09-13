#!/usr/bin/env python3
"""Tests for source and live validation of the [05-UNS-5] issue manifest."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from capacity_census_liveness import IssueKind, IssueRecord, IssueState
from generate_rejection_registries import MANIFEST_REL, load_issue_manifest
from validate_rejection_issue_manifest import adjudicate, main

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
    def test_main_fetches_every_standing_row(self, fetch_issue: mock.Mock) -> None:
        fetch_issue.return_value = IssueRecord(IssueKind.ISSUE, IssueState.OPEN)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)
        expected = load_issue_manifest(ROOT / MANIFEST_REL)
        self.assertEqual(
            [call.args[0] for call in fetch_issue.call_args_list],
            expected,
        )


if __name__ == "__main__":
    unittest.main()
