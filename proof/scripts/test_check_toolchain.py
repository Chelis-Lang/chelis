"""Tests for check_toolchain.py.

Run with: python3 -m unittest proof/scripts/test_check_toolchain.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import check_toolchain as ct  # noqa: E402


class TestParseVersion(unittest.TestCase):
    def test_plain_triple(self):
        self.assertEqual(ct.parse_version("2.7.4"), (2, 7, 4))

    def test_with_prefix(self):
        self.assertEqual(ct.parse_version("vibe 2.7.4"), (2, 7, 4))

    def test_embedded(self):
        self.assertEqual(
            ct.parse_version("Lean (version 4.29.0, x86_64-...)"), (4, 29, 0)
        )

    def test_no_match(self):
        self.assertIsNone(ct.parse_version("no version here"))

    def test_version_tuple_comparable(self):
        self.assertLess(ct.parse_version("2.4.9"), ct.MIN_VIBE_VERSION)
        self.assertGreaterEqual(ct.parse_version("2.7.4"), ct.MIN_VIBE_VERSION)


class TestPinnedToolchainFile(unittest.TestCase):
    """The repo ships proof/lean/lean-toolchain with the pinned version."""

    def test_file_exists_and_matches(self):
        result = ct.check_pinned_toolchain_file()
        self.assertTrue(result.ok, f"unexpected failure: {result.detail}")
        self.assertIn(f"v{ct.PINNED_LEAN_VERSION}", result.detail)


class TestRunAllSmoke(unittest.TestCase):
    """run_all must execute every check without raising, regardless of env."""

    def test_all_checks_return_results(self):
        results = ct.run_all()
        self.assertEqual(len(results), len(ct.CHECKS))
        for r in results:
            self.assertIsInstance(r, ct.CheckResult)
            self.assertIsInstance(r.ok, bool)
            self.assertIsInstance(r.detail, str)
            self.assertTrue(r.name)


if __name__ == "__main__":
    unittest.main()
