#!/usr/bin/env python3
"""Tests for the compensating rejection-authority closure guard."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from generate_rejection_registries import AuthoritySite, RegistryError
from reopen_rejection_authority_issue import (
    guard_closed_issue,
    read_closed_issue_event,
)


class ClosedIssueEvent(unittest.TestCase):
    def test_reads_closed_issue_number(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            event = Path(raw) / "event.json"
            event.write_text(json.dumps({"action": "closed", "issue": {"number": 729}}))
            self.assertEqual(read_closed_issue_event(event), 729)

    def test_rejects_non_issue_or_non_closed_events(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            event = Path(raw) / "event.json"
            for payload in (
                {"action": "opened", "issue": {"number": 729}},
                {"action": "closed", "issue": {"number": 0}},
                {
                    "action": "closed",
                    "issue": {"number": 729, "pull_request": {"url": "pr"}},
                },
            ):
                event.write_text(json.dumps(payload))
                with self.assertRaises(RuntimeError):
                    read_closed_issue_event(event)


class ClosureGuard(unittest.TestCase):
    def test_issue_without_authority_remains_closed(self) -> None:
        calls: list[list[str]] = []

        def run(args: list[str], **_: object) -> object:
            calls.append(args)
            raise AssertionError("GitHub must not be queried without an authority")

        self.assertFalse(
            guard_closed_issue(
                Path("/repo"),
                729,
                discover=lambda _: {714: [AuthoritySite("crate.rs", 1)]},
                run=run,
            )
        )
        self.assertEqual(calls, [])

    def test_closed_issue_with_authority_is_reopened_and_explained(self) -> None:
        calls: list[list[str]] = []

        def run(args: list[str], **_: object) -> object:
            calls.append(args)
            if len(calls) == 1:
                stdout = '{"state":"closed","number":729}'
            elif len(calls) == 2:
                stdout = '{"state":"open","number":729}'
            else:
                stdout = "{}"
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": stdout, "stderr": ""},
            )()

        reopened = guard_closed_issue(
            Path("/repo"),
            729,
            discover=lambda _: {
                729: [AuthoritySite("crates/backend/src/emit.rs", 17)]
            },
            run=run,
        )
        self.assertTrue(reopened)
        self.assertEqual(len(calls), 3)
        self.assertIn("repos/Chelis-Lang/chelis/issues/729", calls[0])
        self.assertIn("PATCH", calls[1])
        self.assertIn("state=open", calls[1])
        self.assertTrue(any("comments" in arg for arg in calls[2]))
        self.assertTrue(
            any("crates/backend/src/emit.rs:17" in arg for arg in calls[2])
        )

    def test_already_reopened_issue_is_idempotent(self) -> None:
        calls: list[list[str]] = []

        def run(args: list[str], **_: object) -> object:
            calls.append(args)
            return type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": '{"state":"open","number":729}',
                    "stderr": "",
                },
            )()

        self.assertFalse(
            guard_closed_issue(
                Path("/repo"),
                729,
                discover=lambda _: {729: [AuthoritySite("crate.rs", 1)]},
                run=run,
            )
        )
        self.assertEqual(len(calls), 1)

    def test_inventory_failure_reopens_fail_closed(self) -> None:
        calls: list[list[str]] = []

        def discover(_: Path) -> dict[int, list[AuthoritySite]]:
            raise RegistryError("unreadable source edge")

        def run(args: list[str], **_: object) -> object:
            calls.append(args)
            if len(calls) == 1:
                stdout = '{"state":"closed","number":729}'
            elif len(calls) == 2:
                stdout = '{"state":"open","number":729}'
            else:
                stdout = "{}"
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": stdout, "stderr": ""},
            )()

        self.assertTrue(
            guard_closed_issue(Path("/repo"), 729, discover=discover, run=run)
        )
        self.assertTrue(any("inventory failed closed" in arg for arg in calls[2]))

    def test_unexpected_inventory_failure_also_reopens_fail_closed(self) -> None:
        calls: list[list[str]] = []

        def discover(_: Path) -> dict[int, list[AuthoritySite]]:
            raise TypeError("unexpected inventory shape")

        def run(args: list[str], **_: object) -> object:
            calls.append(args)
            if len(calls) == 1:
                stdout = '{"state":"closed","number":729}'
            elif len(calls) == 2:
                stdout = '{"state":"open","number":729}'
            else:
                stdout = "{}"
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": stdout, "stderr": ""},
            )()

        self.assertTrue(
            guard_closed_issue(Path("/repo"), 729, discover=discover, run=run)
        )
        self.assertTrue(any("inventory failed closed" in arg for arg in calls[2]))

    def test_malformed_inventory_results_also_reopen_fail_closed(self) -> None:
        malformed = (
            None,
            {729: []},
            {714: []},
            {729: None},
            {729: [object()]},
            {714: [object()]},
            {
                729: [
                    AuthoritySite("z.rs", 2),
                    AuthoritySite("a.rs", 1),
                ]
            },
            {
                729: [
                    AuthoritySite("a.rs", 1),
                    AuthoritySite("a.rs", 1),
                ]
            },
        )
        for result in malformed:
            with self.subTest(result=result):
                calls: list[list[str]] = []

                def run(args: list[str], **_: object) -> object:
                    calls.append(args)
                    if len(calls) == 1:
                        stdout = '{"state":"closed","number":729}'
                    elif len(calls) == 2:
                        stdout = '{"state":"open","number":729}'
                    else:
                        stdout = "{}"
                    return type(
                        "Completed",
                        (),
                        {"returncode": 0, "stdout": stdout, "stderr": ""},
                    )()

                self.assertTrue(
                    guard_closed_issue(
                        Path("/repo"),
                        729,
                        discover=lambda _, result=result: result,  # type: ignore[arg-type]
                        run=run,
                    )
                )
                self.assertTrue(
                    any("inventory failed closed" in arg for arg in calls[2])
                )

    def test_github_failure_does_not_report_success(self) -> None:
        def run(_: list[str], **__: object) -> object:
            return type(
                "Completed",
                (),
                {"returncode": 1, "stdout": "", "stderr": "network down"},
            )()

        with self.assertRaisesRegex(RuntimeError, "network down"):
            guard_closed_issue(
                Path("/repo"),
                729,
                discover=lambda _: {729: [AuthoritySite("crate.rs", 1)]},
                run=run,
            )


if __name__ == "__main__":
    unittest.main()
