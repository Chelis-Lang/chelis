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


class TestScoreOneControl(unittest.TestCase):
    """Test obligation 2 logic with mocked subprocess."""

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


class TestStampPassObligation(unittest.TestCase):
    """Test obligation 3 logic with mocked subprocess."""

    @patch("subprocess.run")
    def test_stamp_tests_green_passes(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=list(oracle.STAMP_NEXTEST_COMMAND),
            returncode=0,
            stdout="12 tests run: 12 passed, 0 failed\n",
            stderr="",
        )
        oracle.check_stamp_pass_integration_tests()

    @patch("subprocess.run")
    def test_stamp_tests_red_fails(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=list(oracle.STAMP_NEXTEST_COMMAND),
            returncode=1,
            stdout="",
            stderr="test failed",
        )
        with self.assertRaises(oracle.OracleFailure):
            oracle.check_stamp_pass_integration_tests()


class TestTopLevelFormRule(unittest.TestCase):
    """Test obligation 3 logic (spec/03 [03-PROG-1]) with mocked runners."""

    def test_rejected_fixtures_carry_the_identification_the_rule_requires(self) -> None:
        self.assertTrue(oracle.TOP_LEVEL_REJECTED_FIXTURES)
        for name, source, identification in oracle.TOP_LEVEL_REJECTED_FIXTURES:
            self.assertTrue(name)
            if identification.startswith("`"):
                # A headed form is identified by its own head, or the
                # obligation would assert against a diagnostic that could not
                # be about this input.
                head = identification.strip("`")
                self.assertTrue(
                    source.startswith(f"({head} ") or source.startswith(f"({head}{{"),
                    f"{name}: `{head}` is not the head of {source!r}",
                )
            else:
                # A headless form is identified by one of the closed classes
                # [03-PROG-2] fixes, never by an ad-hoc phrase.
                self.assertIn(
                    identification,
                    oracle.TOP_LEVEL_HEADLESS_CLASSES,
                    f"{name}: {identification!r} is not a [03-PROG-2] class",
                )

    def test_every_headless_class_has_a_coverage_fixture(self) -> None:
        # The rule's class set is closed; the oracle must walk all of it.
        covered = {
            identification for _, _, identification in oracle.TOP_LEVEL_REJECTED_FIXTURES
        }
        for spelling in oracle.TOP_LEVEL_HEADLESS_CLASSES:
            self.assertIn(spelling, covered, f"{spelling} has no oracle fixture")

    def test_headless_fixtures_are_not_headed_forms(self) -> None:
        # A negative control on the table: a headless row must not be a
        # `(tag ...)` node, or it would be exercising the headed half under a
        # class label.
        for name, source, identification in oracle.TOP_LEVEL_REJECTED_FIXTURES:
            if identification.startswith("`"):
                continue
            stripped = source.strip()
            headed = (
                stripped.startswith("(")
                and len(stripped) > 1
                and (stripped[1].isalpha() or stripped[1] == "_")
            )
            self.assertFalse(headed, f"{name}: {source!r} has a head")

    def test_accepted_fixtures_are_modules_or_declarations(self) -> None:
        admissible = (
            "module",
            "def",
            "defsig",
            "deftype",
            "typealias",
            "defdim",
            "import",
            "import-all",
            "export",
        )
        self.assertTrue(oracle.TOP_LEVEL_ACCEPTED_FIXTURES)
        for name, source in oracle.TOP_LEVEL_ACCEPTED_FIXTURES:
            for line in source.splitlines():
                head = line[1:].split()[0].split("{")[0]
                self.assertIn(head, admissible, f"{name}: {line!r}")

    def _rejection(self, message: str) -> subprocess.CompletedProcess[str]:
        report = {
            "score": 0,
            "errors": [{"kind": "Other", "message": message, "severity": 0.5}],
        }
        return subprocess.CompletedProcess(
            args=["chelis", "check", "x.dp"],
            returncode=2,
            stdout=json.dumps(report),
            stderr="",
        )

    def _accepted(self) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["chelis", "check", "x.dp"],
            returncode=0,
            stdout='{"score": 1, "errors": []}',
            stderr="",
        )

    def _identifications(self) -> dict[str, str]:
        return {
            source: identification
            for _, source, identification in oracle.TOP_LEVEL_REJECTED_FIXTURES
        }

    @patch("unrepresentable_domain_oracle.run_chelis_check")
    def test_a_rejection_identifying_the_form_passes(self, mock_check: MagicMock) -> None:
        identifications = self._identifications()

        def respond(fixture_path: Path) -> subprocess.CompletedProcess[str]:
            identification = identifications.get(Path(fixture_path).read_text())
            if identification is not None:
                return self._rejection(
                    f"stamp error: expected declaration, got {identification}"
                )
            return self._accepted()

        mock_check.side_effect = respond
        oracle.check_top_level_form_rule()

    @patch("unrepresentable_domain_oracle.run_chelis_check")
    def test_an_accepted_non_declaration_fails(self, mock_check: MagicMock) -> None:
        """[03-PROG-1] rejects these; reporting exit 0 is an oracle failure."""
        mock_check.return_value = self._accepted()
        with self.assertRaises(oracle.OracleFailure):
            oracle.check_top_level_form_rule()

    @patch("unrepresentable_domain_oracle.run_chelis_check")
    def test_a_rejection_that_does_not_identify_the_form_fails(
        self, mock_check: MagicMock
    ) -> None:
        """[03-PROG-2] requires the diagnostic to identify the form."""
        mock_check.return_value = self._rejection("stamp error: something went wrong")
        with self.assertRaises(oracle.OracleFailure):
            oracle.check_top_level_form_rule()

    @patch("unrepresentable_domain_oracle.run_chelis_check")
    def test_a_placeholder_identification_fails(self, mock_check: MagicMock) -> None:
        """[03-PROG-2] forbids substituting a placeholder.

        This is the exact regression the red team found: the implementation
        leaked `<non-list>` and `<non-symbol>` while the oracle stayed green
        because its table only covered headed forms.
        """
        identifications = self._identifications()

        def respond(fixture_path: Path) -> subprocess.CompletedProcess[str]:
            source = Path(fixture_path).read_text()
            identification = identifications.get(source)
            if identification is None:
                return self._accepted()
            if identification.startswith("`"):
                return self._rejection(
                    f"stamp error: expected declaration, got {identification}"
                )
            return self._rejection("stamp error: expected declaration, got `<non-list>`")

        mock_check.side_effect = respond
        with self.assertRaises(oracle.OracleFailure):
            oracle.check_top_level_form_rule()

    @patch("unrepresentable_domain_oracle.run_chelis_check")
    def test_a_rejected_admissible_form_fails(self, mock_check: MagicMock) -> None:
        """A `module` wrapper or declaration must not be rejected."""
        identifications = self._identifications()

        def respond(fixture_path: Path) -> subprocess.CompletedProcess[str]:
            identification = identifications.get(Path(fixture_path).read_text())
            if identification is not None:
                return self._rejection(
                    f"stamp error: expected declaration, got {identification}"
                )
            # Every admissible form is rejected too; the obligation must
            # notice rather than reporting a pass.
            return self._rejection("stamp error: expected declaration, got `def`")

        mock_check.side_effect = respond
        with self.assertRaises(oracle.OracleFailure):
            oracle.check_top_level_form_rule()


class TestCompilerApiIngressObligation(unittest.TestCase):
    """Test obligation 5 (chelis#1088) with mocked command execution."""

    @patch("subprocess.run")
    def test_green_suite_passes(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=list(oracle.COMPILER_API_INGRESS_NEXTEST_COMMAND),
            returncode=0,
            stdout="Summary [ 5.2s] 17 tests run: 17 passed, 0 skipped",
            stderr="",
        )
        oracle.check_compiler_api_ingress()

    @patch("subprocess.run")
    def test_red_suite_fails(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=list(oracle.COMPILER_API_INGRESS_NEXTEST_COMMAND),
            returncode=1,
            stdout="",
            stderr="test failed",
        )
        with self.assertRaises(oracle.OracleFailure):
            oracle.check_compiler_api_ingress()

    def test_command_drives_the_compiled_parity_suite(self) -> None:
        cmd = oracle.COMPILER_API_INGRESS_NEXTEST_COMMAND
        self.assertEqual(cmd[0], "cargo")
        self.assertIn("nextest", cmd)
        self.assertIn("chelis-compiler-api", cmd)
        self.assertIn("phase3_stamped_ingress", cmd)


class TestObligationRoster(unittest.TestCase):
    """The roster main() runs is the contract this oracle advertises."""

    def test_every_obligation_is_registered(self) -> None:
        self.assertEqual(
            list(oracle.OBLIGATIONS),
            [
                oracle.check_keyword_in_expr_rejected,
                oracle.check_score_one_controls,
                oracle.check_top_level_form_rule,
                oracle.check_stamp_pass_integration_tests,
                oracle.check_compiler_api_ingress,
            ],
        )


class TestBinaryResolution(unittest.TestCase):
    """The build-once fast path must degrade to `cargo run`, never to a pass."""

    def setUp(self) -> None:
        oracle._BINARY_RESOLUTION_ATTEMPTED = False
        oracle._RESOLVED_CHELIS_BINARY = None

    def tearDown(self) -> None:
        oracle._BINARY_RESOLUTION_ATTEMPTED = False
        oracle._RESOLVED_CHELIS_BINARY = None

    @patch("subprocess.run")
    def test_a_failed_build_falls_back_rather_than_resolving(
        self, mock_run: MagicMock
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["cargo", "build"], returncode=101, stdout="", stderr="build failed"
        )
        self.assertIsNone(oracle.resolve_chelis_binary())

    @patch("subprocess.run")
    def test_resolution_is_attempted_once(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["cargo", "build"], returncode=101, stdout="", stderr=""
        )
        oracle.resolve_chelis_binary()
        oracle.resolve_chelis_binary()
        self.assertEqual(mock_run.call_count, 1)


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
