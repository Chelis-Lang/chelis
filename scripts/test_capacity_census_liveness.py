#!/usr/bin/env python3
"""Unit tests for the capacity-census liveness gate's pure verdict logic.

Run: .venv/bin/python scripts/test_capacity_census_liveness.py
"""

from __future__ import annotations

import unittest

from capacity_census_liveness import adjudicate, extract_issue_refs


def row(citation: str, row_id: str = "x.h: void f(void);") -> dict:
    return {"kind": "header-export", "id": row_id, "citation": citation}


class ExtractIssueRefs(unittest.TestCase):
    def test_extracts_and_dedupes_in_order(self) -> None:
        refs = extract_issue_refs("seam unwinds per chelis#893/chelis#894; see chelis#893")
        self.assertEqual(refs, [893, 894])

    def test_baseline_tag_has_no_refs(self) -> None:
        self.assertEqual(extract_issue_refs("baseline-2026-07-30"), [])


class Adjudicate(unittest.TestCase):
    def test_open_citation_passes(self) -> None:
        problems = adjudicate([row("chelis#893")], {893: "OPEN"})
        self.assertEqual(problems, [])

    def test_baseline_only_citation_passes_without_lookup(self) -> None:
        problems = adjudicate([row("baseline-2026-07-30")], {})
        self.assertEqual(problems, [])

    def test_closed_citation_fails_with_readjudication_message(self) -> None:
        problems = adjudicate([row("chelis#893")], {893: "CLOSED"})
        self.assertEqual(len(problems), 1)
        self.assertIn("STALE citation", problems[0])
        self.assertIn("re-adjudicated", problems[0])

    def test_one_closed_ref_among_open_still_fails(self) -> None:
        problems = adjudicate(
            [row("chelis#893 and chelis#894")], {893: "OPEN", 894: "CLOSED"}
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


if __name__ == "__main__":
    unittest.main()
