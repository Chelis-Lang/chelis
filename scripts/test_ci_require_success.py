#!/usr/bin/env python3
"""Unit tests for the required-job result aggregator."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

import ci_require_success


class ParseResultTests(unittest.TestCase):
    def test_accepts_named_success_results(self) -> None:
        self.assertEqual(
            ci_require_success.parse_results(["workspace=success", "oracle=success"]),
            [("workspace", "success"), ("oracle", "success")],
        )

    def test_rejects_an_argument_without_a_job_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "NAME=RESULT"):
            ci_require_success.parse_results(["success"])


class MainTests(unittest.TestCase):
    def test_all_successful_dependencies_pass(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                ci_require_success.main(["workspace=success", "oracle=success"]),
                0,
            )
        self.assertIn("workspace: success", output.getvalue())

    def test_failed_dependency_fails_the_aggregate(self) -> None:
        errors = io.StringIO()
        with redirect_stderr(errors):
            self.assertEqual(
                ci_require_success.main(["workspace=failure", "oracle=success"]),
                1,
            )
        self.assertIn("workspace: failure", errors.getvalue())

    def test_skipped_dependency_fails_the_aggregate(self) -> None:
        errors = io.StringIO()
        with redirect_stderr(errors):
            self.assertEqual(
                ci_require_success.main(["workspace=success", "oracle=skipped"]),
                1,
            )


if __name__ == "__main__":
    unittest.main()
