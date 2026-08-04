#!/usr/bin/env python3
"""Tests for live validation of the [05-UNS-5] issue manifest."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from capacity_census_liveness import IssueKind, IssueRecord, IssueState
from validate_rejection_issue_manifest import (
    ClosingSource,
    adjudicate,
    adjudicate_closing_references,
    collect_pull_request_sources,
    find_closing_references,
)


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


class ClosingReferences(unittest.TestCase):
    def test_detects_github_closing_forms_including_fixed_colon(self) -> None:
        sources = [
            ClosingSource("pull request body", "Part of #730. Fixed: #729 Phase 2"),
            ClosingSource(
                "commit abc123",
                "Resolves Chelis-Lang/chelis#714 and closes "
                "https://github.com/Chelis-Lang/chelis/issues/879",
            ),
        ]
        self.assertEqual(find_closing_references(sources), {714, 729, 879})

    def test_nonclosing_and_historical_text_do_not_count(self) -> None:
        sources = [
            ClosingSource(
                "pull request body",
                "Part of #729; historical discussion mentions #691",
            ),
        ]
        self.assertEqual(find_closing_references(sources), set())

    def test_closing_live_authority_reports_every_construction_site(self) -> None:
        rows = [
            {
                "number": 714,
                "sites": [
                    {"path": "crates/backend/src/emit.rs", "line": 17},
                    {"path": "crates/backend/src/host.rs", "line": 23},
                ],
            }
        ]
        problems = adjudicate_closing_references(
            rows,
            [ClosingSource("commit abc123", "Fixes #714")],
        )
        self.assertEqual(len(problems), 2)
        self.assertIn("crates/backend/src/emit.rs:17", problems[0])
        self.assertIn("crates/backend/src/host.rs:23", problems[1])
        self.assertTrue(all("commit abc123" in problem for problem in problems))

    def test_closing_issue_with_no_executable_authority_passes(self) -> None:
        self.assertEqual(
            adjudicate_closing_references(
                [{"number": 714, "sites": []}],
                [ClosingSource("pull request body", "Closes #714")],
            ),
            [],
        )

    def test_collects_only_pr_merge_inputs_and_all_commit_pages(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            event_path = Path(raw) / "event.json"
            event_path.write_text(
                json.dumps(
                    {
                        "number": 42,
                        "pull_request": {
                            "number": 42,
                            "title": "Part of #729",
                            "body": "Closes #1143",
                        },
                    }
                )
            )

            def run(args: list[str], **_: object) -> object:
                self.assertIn("pulls/42/commits?per_page=100", args[-1])
                return type(
                    "Completed",
                    (),
                    {
                        "returncode": 0,
                        "stderr": "",
                        "stdout": json.dumps(
                            [
                                [
                                    {
                                        "sha": "abcdef0123456789",
                                        "commit": {"message": "Fixes #691"},
                                    }
                                ],
                                [
                                    {
                                        "sha": "9876543210abcdef",
                                        "commit": {"message": "Part of #729"},
                                    }
                                ],
                            ]
                        ),
                    },
                )()

            sources = collect_pull_request_sources(event_path, run=run)
        self.assertEqual(find_closing_references(sources), {691, 1143})
        self.assertEqual(
            [source.label for source in sources],
            [
                "pull request title",
                "pull request body",
                "commit abcdef012345",
                "commit 9876543210ab",
            ],
        )

    def test_commit_fetch_failure_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            event_path = Path(raw) / "event.json"
            event_path.write_text(
                '{"number":42,"pull_request":{"number":42}}'
            )

            def run(_: list[str], **__: object) -> object:
                return type(
                    "Completed",
                    (),
                    {"returncode": 1, "stderr": "network down", "stdout": ""},
                )()

            with self.assertRaisesRegex(RuntimeError, "network down"):
                collect_pull_request_sources(event_path, run=run)


if __name__ == "__main__":
    unittest.main()
