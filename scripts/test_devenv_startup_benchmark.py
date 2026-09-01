#!/usr/bin/env python3
"""Tests for Devenv startup timing decomposition."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from devenv_startup_benchmark import Measurement, normalize_payload, summarize


class StartupBenchmarkSummaryTests(unittest.TestCase):
    def test_summary_keeps_wall_and_cpu_separate(self) -> None:
        summary = summarize(
            [
                Measurement(wall_s=1.8, user_s=0.1, system_s=0.05),
                Measurement(wall_s=1.2, user_s=0.2, system_s=0.03),
                Measurement(wall_s=1.5, user_s=0.15, system_s=0.04),
            ]
        )
        self.assertEqual(summary["wall_median_s"], 1.5)
        self.assertEqual(summary["user_median_s"], 0.15)
        self.assertEqual(summary["system_median_s"], 0.04)

    def test_summary_rejects_an_empty_sample(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            summarize([])

    def test_payload_separator_is_not_executed_as_a_command(self) -> None:
        self.assertEqual(normalize_payload(["--", "python", "-c", "pass"]), ["python", "-c", "pass"])
        self.assertEqual(normalize_payload(["python", "-c", "pass"]), ["python", "-c", "pass"])


if __name__ == "__main__":
    unittest.main()
