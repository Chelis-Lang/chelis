#!/usr/bin/env python3
"""Unit tests for scripts/unrepresentable_domain_oracle.py.

Tests the oracle's logic (fixture validation, JSON parsing, error detection)
without requiring a full cargo build. Integration-level tests that actually
invoke `chelis check` are covered by running the oracle itself.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure the scripts directory is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import unrepresentable_domain_oracle as oracle


class TestFixtureInventory(unittest.TestCase):
    """Verify the fixture lists are non-empty and well-formed."""

    def test_keyword_fixtures_non_empty(self) -> None:
        self.assertGreater(len(oracle.KEYWORD_IN_EXPR_FIXTURES), 0)

    def test_bare_name_fixtures_non_empty(self) -> None:
        self.assertGreater(len(oracle.BARE_NAME_IN_EXPR_FIXTURES), 0)

    def test_keyword_fixtures_have_names_and_sources(self) -> None:
        for name, source in oracle.KEYWORD_IN_EXPR_FIXTURES:
            self.assertIsInstance(name, str)
            self.assertIsInstance(source, str)
            self.assertIn(":", source, f"fixture '{name}' should contain a keyword")

    def test_score_one_fixtures_non_empty(self) -> None:
        self.assertGreater(len(oracle.SCORE_ONE_CONTROL_FIXTURES), 0)

    def test_score_one_fixtures_have_names_and_sources(self) -> None:
        for name, source in oracle.SCORE_ONE_CONTROL_FIXTURES:
            self.assertIsInstance(name, str)
            self.assertIsInstance(source, str)
            # All should be valid Deep s-expressions starting with (
            self.assertTrue(
                source.startswith("("),
                f"fixture '{name}' should be a Deep s-expression",
            )

    def test_score_one_fixtures_have_no_bare_keywords(self) -> None:
        """Control fixtures should not contain bare :keywords in expr slots."""
        for name, source in oracle.SCORE_ONE_CONTROL_FIXTURES:
            # These are structural-name programs; they should not have bare
            # :keywords outside metadata maps.
            # A simple heuristic: no `:` outside `{}` braces at the top level
            # of any child position. This is a structural check, not a parser.
            self.assertNotIn(
                " :",
                source.replace("{}", ""),
                f"fixture '{name}' should not have bare keywords in expr slots",
            )


class TestParseCheckJson(unittest.TestCase):
    """Test the JSON parsing helper."""

    def test_valid_json(self) -> None:
        report = oracle.parse_check_json('{"score": 1, "errors": []}')
        self.assertEqual(report["score"], 1)
        self.assertEqual(report["errors"], [])

    def test_invalid_json_raises(self) -> None:
        with self.assertRaises(oracle.OracleFailure):
            oracle.parse_check_json("not json at all")

    def test_empty_string_raises(self) -> None:
        with self.assertRaises(oracle.OracleFailure):
            oracle.parse_check_json("")


class TestWriteFixture(unittest.TestCase):
    """Test the temp-file fixture writer."""

    def test_creates_file_with_content(self) -> None:
        content = "(def {} f (lit {} 1))"
        path = oracle.write_fixture(content)
        try:
            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(), content)
            self.assertTrue(path.name.endswith(".dp"))
        finally:
            path.unlink(missing_ok=True)

    def test_custom_suffix(self) -> None:
        path = oracle.write_fixture("x", suffix=".ch")
        try:
            self.assertTrue(path.name.endswith(".ch"))
        finally:
            path.unlink(missing_ok=True)


class TestKeywordRejection(unittest.TestCase):
    """Test obligation 1 logic with mocked subprocess."""

    def _mock_check_error(self, source: str) -> subprocess.CompletedProcess[str]:
        """Simulate chelis check returning a keyword parse error."""
        report = {
            "score": 0,
            "components": {"parse": 0, "structure": 0, "names": 0, "types": 0},
            "typed_nodes": 0,
            "untyped_nodes": 0,
            "total_nodes": 0,
            "unresolved_names": [],
            "errors": [
                {
                    "kind": "Other",
                    "message": "expected expression (bare :keyword is valid only as a metadata map key), found Keyword(\"bad\") at byte 10",
                    "severity": 0.5,
                }
            ],
        }
        return subprocess.CompletedProcess(
            args=["chelis", "check", "x.dp"],
            returncode=2,
            stdout=json.dumps(report),
            stderr="",
        )

    @patch("unrepresentable_domain_oracle.run_chelis_check")
    def test_keyword_rejected_passes(self, mock_check: MagicMock) -> None:
        mock_check.side_effect = self._mock_check_error
        # Should not raise.
        oracle.check_keyword_in_expr_rejected()

    @patch("unrepresentable_domain_oracle.run_chelis_check")
    def test_keyword_accepted_fails(self, mock_check: MagicMock) -> None:
        """If chelis check returns exit 0 for a keyword fixture, oracle fails."""
        mock_check.return_value = subprocess.CompletedProcess(
            args=["chelis", "check", "x.dp"],
            returncode=0,
            stdout='{"score": 1, "errors": []}',
            stderr="",
        )
        with self.assertRaises(oracle.OracleFailure):
            oracle.check_keyword_in_expr_rejected()

    @patch("unrepresentable_domain_oracle.run_chelis_check")
    def test_keyword_error_without_keyword_mention_fails(
        self, mock_check: MagicMock
    ) -> None:
        """Error must mention 'keyword' to pass."""
        report = {
            "score": 0,
            "errors": [{"kind": "Other", "message": "some other error", "severity": 0.5}],
        }
        mock_check.return_value = subprocess.CompletedProcess(
            args=["chelis", "check", "x.dp"],
            returncode=2,
            stdout=json.dumps(report),
            stderr="",
        )
        with self.assertRaises(oracle.OracleFailure):
            oracle.check_keyword_in_expr_rejected()


class TestBareNameRejection(unittest.TestCase):
    """The CLI must expose the stamped RuntimeExpr ingress."""

    @patch("unrepresentable_domain_oracle.run_chelis_check")
    def test_bare_name_stamp_error_passes(self, mock_check: MagicMock) -> None:
        report = {
            "score": 0,
            "errors": [
                {
                    "kind": "Other",
                    "message": "stamp error: bare name `x` at expression slot",
                    "severity": 1.0,
                }
            ],
        }
        mock_check.return_value = subprocess.CompletedProcess(
            args=["chelis", "check", "x.dp"],
            returncode=2,
            stdout=json.dumps(report),
            stderr="",
        )
        oracle.check_bare_name_in_expr_rejected()

    @patch("unrepresentable_domain_oracle.run_chelis_check")
    def test_legacy_acceptance_fails(self, mock_check: MagicMock) -> None:
        mock_check.return_value = subprocess.CompletedProcess(
            args=["chelis", "check", "x.dp"],
            returncode=0,
            stdout='{"score": 1, "errors": []}',
            stderr="",
        )
        with self.assertRaises(oracle.OracleFailure):
            oracle.check_bare_name_in_expr_rejected()


class TestScoreOneControl(unittest.TestCase):
    """Test obligation 3 logic with mocked subprocess."""

    def _mock_score_one(self, fixture_path: Path) -> subprocess.CompletedProcess[str]:
        report = {
            "score": 1,
            "components": {"parse": 1, "structure": 1, "names": 1, "types": 1},
            "typed_nodes": 2,
            "untyped_nodes": 0,
            "total_nodes": 2,
            "unresolved_names": [],
            "errors": [],
        }
        return subprocess.CompletedProcess(
            args=["chelis", "check", str(fixture_path)],
            returncode=0,
            stdout=json.dumps(report),
            stderr="",
        )

    @patch("unrepresentable_domain_oracle.run_chelis_check")
    def test_score_one_passes(self, mock_check: MagicMock) -> None:
        mock_check.side_effect = self._mock_score_one
        oracle.check_score_one_controls()

    @patch("unrepresentable_domain_oracle.run_chelis_check")
    def test_score_below_one_fails(self, mock_check: MagicMock) -> None:
        report = {
            "score": 0.8,
            "errors": [{"kind": "Type", "message": "unresolved name", "severity": 0.2}],
        }
        mock_check.return_value = subprocess.CompletedProcess(
            args=["chelis", "check", "x.dp"],
            returncode=0,
            stdout=json.dumps(report),
            stderr="",
        )
        with self.assertRaises(oracle.OracleFailure):
            oracle.check_score_one_controls()

    @patch("unrepresentable_domain_oracle.run_chelis_check")
    def test_nonzero_exit_fails(self, mock_check: MagicMock) -> None:
        mock_check.return_value = subprocess.CompletedProcess(
            args=["chelis", "check", "x.dp"],
            returncode=1,
            stdout="",
            stderr="error",
        )
        with self.assertRaises(oracle.OracleFailure):
            oracle.check_score_one_controls()


class TestSuccessorSuiteObligation(unittest.TestCase):
    """Test the stamped-ingress and successor-carrier suite command."""

    @patch("subprocess.run")
    def test_stamp_tests_green_passes(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=list(oracle.SUCCESSOR_NEXTEST_COMMAND),
            returncode=0,
            stdout="12 tests run: 12 passed, 0 failed\n",
            stderr="",
        )
        oracle.check_successor_integration_tests()

    @patch("subprocess.run")
    def test_stamp_tests_red_fails(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=list(oracle.SUCCESSOR_NEXTEST_COMMAND),
            returncode=1,
            stdout="",
            stderr="test failed",
        )
        with self.assertRaises(oracle.OracleFailure):
            oracle.check_successor_integration_tests()


class TestChelisCheckCommand(unittest.TestCase):
    """Test the command builder."""

    def test_command_shape(self) -> None:
        cmd = oracle.chelis_check_command()
        self.assertEqual(cmd[0], "cargo")
        self.assertIn("chelis-cli", cmd)
        self.assertIn("check", cmd)
        self.assertIn("--quiet", cmd)
        self.assertIn("--allow-style-violations", cmd)


class TestRepoRoot(unittest.TestCase):
    """Test REPO_ROOT resolution."""

    def test_repo_root_exists(self) -> None:
        self.assertTrue(oracle.REPO_ROOT.is_dir())

    def test_repo_root_has_cargo_toml(self) -> None:
        self.assertTrue((oracle.REPO_ROOT / "Cargo.toml").is_file())


if __name__ == "__main__":
    unittest.main()
