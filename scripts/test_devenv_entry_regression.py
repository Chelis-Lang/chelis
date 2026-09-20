#!/usr/bin/env python3
"""Tests for concurrent Devenv entry result classification."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from devenv_entry_regression import EntryResult, classify_entry


class ConcurrentEntryClassificationTests(unittest.TestCase):
    def test_successful_payload_is_accepted(self) -> None:
        result = EntryResult(0, "Running tasks in 21.4s\nPAYLOAD_3\n", "PAYLOAD_3")
        self.assertIsNone(classify_entry(result))

    def test_failed_entry_must_not_run_payload(self) -> None:
        result = EntryResult(1, "dependency failed\n", "PAYLOAD_3")
        self.assertIsNone(classify_entry(result))

    def test_fail_open_result_is_rejected(self) -> None:
        result = EntryResult(
            0,
            "devenv:enterShell failed\nPAYLOAD_3\n",
            "PAYLOAD_3",
        )
        self.assertIn("fail-open", classify_entry(result) or "")

    def test_failed_task_summary_cannot_be_mistaken_for_success(self) -> None:
        result = EntryResult(0, "Running tasks in 2s (failed)\nPAYLOAD_3\n", "PAYLOAD_3")
        self.assertIn("fail-open", classify_entry(result) or "")

    def test_zero_without_payload_is_rejected(self) -> None:
        result = EntryResult(0, "tasks complete\n", "PAYLOAD_3")
        self.assertIn("without payload", classify_entry(result) or "")

    def test_missing_exports_race_is_always_rejected(self) -> None:
        result = EntryResult(
            1,
            "chmod: cannot access '.devenv/load-exports': No such file or directory\n",
            "PAYLOAD_3",
        )
        self.assertIn("load-exports", classify_entry(result) or "")


if __name__ == "__main__":
    unittest.main()
