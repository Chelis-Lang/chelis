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
    adjudicate_closing_issue_numbers,
    adjudicate_closing_references,
    collect_pull_request_closing_issues,
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

    def test_manually_linked_live_authority_fails_without_keywords(self) -> None:
        problems = adjudicate_closing_issue_numbers(
            [
                {
                    "number": 729,
                    "sites": [{"path": "crates/backend/src/emit.rs", "line": 17}],
                }
            ],
            {729},
            "GitHub closingIssuesReferences",
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("crates/backend/src/emit.rs:17", problems[0])
        self.assertIn("GitHub closingIssuesReferences", problems[0])

    def test_collects_only_pr_merge_inputs_and_all_commit_pages(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            event_path = Path(raw) / "event.json"
            event_path.write_text(
                json.dumps(
                    {
                        "number": 42,
                        "pull_request": {
                            "number": 42,
                            "commits": 2,
                            "head": {
                                "sha": "9876543210abcdef9876543210abcdef98765432"
                            },
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
                                        "sha": "abcdef0123456789abcdef0123456789abcdef01",
                                        "commit": {"message": "Fixes #691"},
                                    }
                                ],
                                [
                                    {
                                        "sha": "9876543210abcdef9876543210abcdef98765432",
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

    def test_squash_merge_title_closing_keyword_is_rejected(self) -> None:
        def run(_: list[str], **__: object) -> object:
            return type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stderr": "",
                    "stdout": json.dumps(
                        [[{"sha": "a" * 40, "commit": {"message": "ordinary"}}]]
                    ),
                },
            )()

        with tempfile.TemporaryDirectory() as raw:
            event_path = Path(raw) / "event.json"
            event_path.write_text(
                json.dumps(
                    {
                        "number": 42,
                        "pull_request": {
                            "number": 42,
                            "commits": 1,
                            "head": {"sha": "a" * 40},
                            "title": "Fixes #729",
                            "body": "Part of #729",
                        },
                    }
                )
            )
            sources = collect_pull_request_sources(event_path, run=run)
        self.assertEqual(find_closing_references(sources), {729})
        self.assertEqual(sources[0].label, "pull request title")

    def test_edited_body_is_rechecked_without_a_head_change(self) -> None:
        def run(_: list[str], **__: object) -> object:
            return type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stderr": "",
                    "stdout": json.dumps(
                        [[{"sha": "a" * 40, "commit": {"message": "ordinary"}}]]
                    ),
                },
            )()

        with tempfile.TemporaryDirectory() as raw:
            event_path = Path(raw) / "event.json"
            event = {
                "action": "edited",
                "number": 42,
                "pull_request": {
                    "number": 42,
                    "commits": 1,
                    "head": {"sha": "a" * 40},
                    "title": "Part of #729",
                    "body": "Closes #729",
                },
            }
            event_path.write_text(json.dumps(event))
            self.assertEqual(
                find_closing_references(collect_pull_request_sources(event_path, run=run)),
                {729},
            )
            event["pull_request"]["body"] = "Part of #729"
            event_path.write_text(json.dumps(event))
            self.assertEqual(
                find_closing_references(collect_pull_request_sources(event_path, run=run)),
                set(),
            )

    def test_commit_fetch_failure_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            event_path = Path(raw) / "event.json"
            event_path.write_text(
                '{"number":42,"pull_request":{"number":42,"commits":1,'
                '"head":{"sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},'
                '"title":"ordinary","body":""}}'
            )

            def run(_: list[str], **__: object) -> object:
                return type(
                    "Completed",
                    (),
                    {"returncode": 1, "stderr": "network down", "stdout": ""},
                )()

            with self.assertRaisesRegex(RuntimeError, "network down"):
                collect_pull_request_sources(event_path, run=run)

    def test_incomplete_commit_inventory_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            event_path = Path(raw) / "event.json"
            event_path.write_text(
                '{"number":42,"pull_request":{"number":42,"commits":251,'
                '"head":{"sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},'
                '"title":"ordinary","body":""}}'
            )

            def run(_: list[str], **__: object) -> object:
                commits = [
                    {"sha": f"{index:040x}", "commit": {"message": "ordinary"}}
                    for index in range(250)
                ]
                return type(
                    "Completed",
                    (),
                    {
                        "returncode": 0,
                        "stderr": "",
                        "stdout": json.dumps([commits]),
                    },
                )()

            with self.assertRaisesRegex(RuntimeError, "expected 251 commits, received 250"):
                collect_pull_request_sources(event_path, run=run)

    def test_malformed_commit_message_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            event_path = Path(raw) / "event.json"
            event_path.write_text(
                '{"number":42,"pull_request":{"number":42,"commits":1,'
                '"head":{"sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},'
                '"title":"ordinary","body":""}}'
            )

            def run(_: list[str], **__: object) -> object:
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
                                        "sha": "a" * 40,
                                        "commit": {"message": None},
                                    }
                                ]
                            ]
                        ),
                    },
                )()

            with self.assertRaisesRegex(RuntimeError, "invalid commit message"):
                collect_pull_request_sources(event_path, run=run)

    def test_malformed_commit_sha_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            event_path = Path(raw) / "event.json"
            event_path.write_text(
                '{"number":42,"pull_request":{"number":42,"commits":1,'
                '"head":{"sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},'
                '"title":"ordinary","body":""}}'
            )

            def run(_: list[str], **__: object) -> object:
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
                                        "sha": "not-a-full-object-id",
                                        "commit": {"message": "ordinary"},
                                    }
                                ]
                            ]
                        ),
                    },
                )()

            with self.assertRaisesRegex(RuntimeError, "invalid or duplicate commit SHA"):
                collect_pull_request_sources(event_path, run=run)

    def test_rest_commit_inventory_must_end_at_the_event_head(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            event_path = Path(raw) / "event.json"
            event_path.write_text(
                json.dumps(
                    {
                        "number": 42,
                        "pull_request": {
                            "number": 42,
                            "commits": 1,
                            "head": {"sha": "a" * 40},
                            "title": "ordinary",
                            "body": "",
                        },
                    }
                )
            )

            def run(_: list[str], **__: object) -> object:
                return type(
                    "Completed",
                    (),
                    {
                        "returncode": 0,
                        "stderr": "",
                        "stdout": json.dumps(
                            [[{"sha": "b" * 40, "commit": {"message": "ordinary"}}]]
                        ),
                    },
                )()

            with self.assertRaisesRegex(RuntimeError, "does not match event head"):
                collect_pull_request_sources(event_path, run=run)

    def test_blank_title_zero_commits_and_malformed_head_fail_closed(self) -> None:
        malformed = (
            {"commits": 1, "head": {"sha": "a" * 40}, "title": "   "},
            {"commits": 0, "head": {"sha": "a" * 40}, "title": "ordinary"},
            {"commits": 1, "head": {"sha": "NOT-A-SHA"}, "title": "ordinary"},
        )
        for pull_request in malformed:
            with self.subTest(pull_request=pull_request), tempfile.TemporaryDirectory() as raw:
                event_path = Path(raw) / "event.json"
                event_path.write_text(
                    json.dumps(
                        {
                            "number": 42,
                            "pull_request": {
                                "number": 42,
                                "body": "",
                                **pull_request,
                            },
                        }
                    )
                )

                def run(_: list[str], **__: object) -> object:
                    self.fail("malformed PR metadata must fail before REST access")

                with self.assertRaisesRegex(RuntimeError, "valid"):
                    collect_pull_request_sources(event_path, run=run)

    def test_collects_manually_linked_closing_issues_across_all_pages(self) -> None:
        responses = iter(
            [
                {
                    "data": {
                        "repository": {
                            "pullRequest": {
                                "closingIssuesReferences": {
                                    "nodes": [
                                        {
                                            "number": 729,
                                            "repository": {"nameWithOwner": "Chelis-Lang/chelis"},
                                        }
                                    ],
                                    "pageInfo": {"hasNextPage": True, "endCursor": "next"},
                                }
                            }
                        }
                    }
                },
                {
                    "data": {
                        "repository": {
                            "pullRequest": {
                                "closingIssuesReferences": {
                                    "nodes": [
                                        {
                                            "number": 1143,
                                            "repository": {"nameWithOwner": "Chelis-Lang/chelis"},
                                        },
                                        {
                                            "number": 9,
                                            "repository": {"nameWithOwner": "someone/else"},
                                        },
                                    ],
                                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                                }
                            }
                        }
                    }
                },
            ]
        )
        calls: list[list[str]] = []

        def run(args: list[str], **_: object) -> object:
            calls.append(args)
            return type(
                "Completed",
                (),
                {"returncode": 0, "stderr": "", "stdout": json.dumps(next(responses))},
            )()

        self.assertEqual(collect_pull_request_closing_issues(42, run=run), {729, 1143})
        self.assertEqual(len(calls), 2)
        self.assertFalse(any("cursor=next" in arg for arg in calls[0]))
        self.assertTrue(any("cursor=next" in arg for arg in calls[1]))

    def test_malformed_closing_issue_graph_fails_closed(self) -> None:
        def run(_: list[str], **__: object) -> object:
            return type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stderr": "",
                    "stdout": '{"data":{"repository":{"pullRequest":null}}}',
                },
            )()

        with self.assertRaisesRegex(RuntimeError, "malformed closing-issue response"):
            collect_pull_request_closing_issues(42, run=run)

    def test_partial_graphql_data_with_errors_fails_closed(self) -> None:
        def run(_: list[str], **__: object) -> object:
            return type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stderr": "",
                    "stdout": json.dumps(
                        {
                            "errors": [{"message": "partial result"}],
                            "data": {
                                "repository": {
                                    "pullRequest": {
                                        "closingIssuesReferences": {
                                            "nodes": [],
                                            "pageInfo": {
                                                "hasNextPage": False,
                                                "endCursor": None,
                                            },
                                        }
                                    }
                                }
                            },
                        }
                    ),
                },
            )()

        with self.assertRaisesRegex(RuntimeError, "GraphQL errors"):
            collect_pull_request_closing_issues(42, run=run)


if __name__ == "__main__":
    unittest.main()
