#!/usr/bin/env python3
"""Unit tests for the capacity-census liveness gate's pure verdict logic.

Run: .venv/bin/python scripts/test_capacity_census_liveness.py
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from capacity_census_liveness import (
    IssueKind,
    IssueRecord,
    IssueState,
    adjudicate,
    extract_issue_refs,
    fetch_issue,
    load_census_rows,
)


def row(citation: str, row_id: str = "x.h: void f(void);") -> dict:
    return {"kind": "header-export", "id": row_id, "citation": citation}


class ExtractIssueRefs(unittest.TestCase):
    def test_extracts_and_dedupes_in_order(self) -> None:
        refs = extract_issue_refs("seam unwinds per chelis#893/chelis#894; see chelis#893")
        self.assertEqual(refs, [893, 894])

    def test_baseline_tag_has_no_refs(self) -> None:
        self.assertEqual(extract_issue_refs("baseline-2026-07-30"), [])


class LoadCensusRows(unittest.TestCase):
    def test_top_level_citation_is_inherited_without_overriding_row_citation(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            census = root / "typed.json"
            census.write_text(
                '{"citation":"chelis#729","rows":['
                '{"kind":"wire","id":"inherited"},'
                '{"kind":"wire","id":"specific","citation":"chelis#893"}'
                "]}"
            )
            rows = load_census_rows(root, (Path("typed.json"),))

        self.assertEqual(rows[0]["citation"], "chelis#729")
        self.assertEqual(rows[1]["citation"], "chelis#893")


class Adjudicate(unittest.TestCase):
    def test_open_citation_passes(self) -> None:
        problems = adjudicate(
            [row("chelis#893")],
            {893: IssueRecord(kind=IssueKind.ISSUE, state=IssueState.OPEN)},
        )
        self.assertEqual(problems, [])

    def test_refless_citation_is_not_liveness_checked(self) -> None:
        # The pure verdict logic is lenient about ref-less citations; the
        # Rust tripwire owns that rejection (CITATION NAMES NO ISSUE), so
        # this documents the division of labor rather than an allowance.
        problems = adjudicate([row("baseline-2026-07-30")], {})
        self.assertEqual(problems, [])

    def test_closed_citation_fails_with_readjudication_message(self) -> None:
        problems = adjudicate(
            [row("chelis#893")],
            {893: IssueRecord(kind=IssueKind.ISSUE, state=IssueState.CLOSED)},
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("STALE citation", problems[0])
        self.assertIn("re-adjudicated", problems[0])

    def test_one_closed_ref_among_open_still_fails(self) -> None:
        problems = adjudicate(
            [row("chelis#893 and chelis#894")],
            {
                893: IssueRecord(kind=IssueKind.ISSUE, state=IssueState.OPEN),
                894: IssueRecord(kind=IssueKind.ISSUE, state=IssueState.CLOSED),
            },
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("chelis#894", problems[0])

    def test_todo_and_empty_citations_fail(self) -> None:
        problems = adjudicate([row("TODO"), row("")], {})
        self.assertEqual(len(problems), 2)
        for problem in problems:
            self.assertIn("UNCITED", problem)

    def test_unresolvable_citation_fails(self) -> None:
        problems = adjudicate([row("chelis#999999")], {})
        self.assertEqual(len(problems), 1)
        self.assertIn("UNRESOLVABLE", problems[0])

    def test_open_pull_request_is_not_an_open_issue(self) -> None:
        problems = adjudicate(
            [row("chelis#956")],
            {956: IssueRecord(kind=IssueKind.PULL_REQUEST, state=IssueState.OPEN)},
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("PULL REQUEST", problems[0])
        self.assertIn("not an OPEN issue", problems[0])


class FetchIssue(unittest.TestCase):
    def test_rest_issue_payload_is_typed(self) -> None:
        calls: list[list[str]] = []

        def run(args: list[str], **_: object) -> object:
            calls.append(args)
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": '{"state":"open","number":729}'},
            )()

        record = fetch_issue(729, run=run)
        self.assertEqual(
            record, IssueRecord(kind=IssueKind.ISSUE, state=IssueState.OPEN)
        )
        self.assertEqual(
            calls,
            [["gh", "api", "repos/Chelis-Lang/chelis/issues/729"]],
            "the REST issues endpoint exposes pull_request identity unlike gh issue view",
        )

    def test_rest_pull_request_payload_is_rejected_by_adjudication(self) -> None:
        def run(_: list[str], **__: object) -> object:
            return type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": (
                        '{"state":"open","number":956,'
                        '"pull_request":{"url":"https://api.github.test/pulls/956"}}'
                    ),
                },
            )()

        record = fetch_issue(956, run=run)
        self.assertEqual(
            record,
            IssueRecord(kind=IssueKind.PULL_REQUEST, state=IssueState.OPEN),
        )
        problems = adjudicate([row("chelis#956")], {956: record})
        self.assertEqual(len(problems), 1)
        self.assertIn("PULL REQUEST", problems[0])

    def test_closed_and_unresolvable_have_negative_parity(self) -> None:
        def closed(_: list[str], **__: object) -> object:
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": '{"state":"closed","number":729}'},
            )()

        def missing(_: list[str], **__: object) -> object:
            return type("Completed", (), {"returncode": 1, "stdout": ""})()

        self.assertEqual(
            fetch_issue(729, run=closed),
            IssueRecord(kind=IssueKind.ISSUE, state=IssueState.CLOSED),
        )
        self.assertIsNone(fetch_issue(999999, run=missing))

    def test_mismatched_or_non_integer_issue_identity_fails_closed(self) -> None:
        payloads = (
            '{"state":"open","number":714}',
            '{"state":"open","number":true}',
            '[{"state":"open","number":729}]',
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                def run(_: list[str], **__: object) -> object:
                    return type(
                        "Completed",
                        (),
                        {"returncode": 0, "stdout": payload},
                    )()

                self.assertIsNone(fetch_issue(729, run=run))


if __name__ == "__main__":
    unittest.main()
