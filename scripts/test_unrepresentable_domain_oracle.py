#!/usr/bin/env python3
"""Unit tests for scripts/unrepresentable_domain_oracle.py.

Tests the oracle's logic (fixture validation, JSON parsing, error detection)
without requiring a full cargo build. Integration-level tests that actually
invoke `chelis check` are covered by running the oracle itself.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
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

    def test_name_fixtures_non_empty(self) -> None:
        self.assertGreater(len(oracle.NAME_IN_EXPR_FIXTURES), 0)

    def test_name_fixtures_cover_the_four_expression_positions(self) -> None:
        """Obligation 7 covers def body, fn body, app argument, bind RHS."""
        names = [name for name, _ in oracle.NAME_IN_EXPR_FIXTURES]
        for position in ("def body", "fn body", "app argument", "bind RHS"):
            self.assertTrue(
                any(position in name for name in names),
                f"no NAME_IN_EXPR fixture covers the {position} position",
            )

    def test_name_fixtures_spell_the_remediation_name(self) -> None:
        """Every fixture uses `oops` so the remediation assertion is exact."""
        for name, source in oracle.NAME_IN_EXPR_FIXTURES:
            self.assertIn(
                "oops",
                source,
                f"fixture '{name}' must use the `oops` identifier the "
                "remediation constant asserts",
            )
        self.assertIn("oops", oracle.NAME_IN_EXPR_REMEDIATION)

    def test_score_one_fixtures_cover_structural_symbol_positions(self) -> None:
        """The chelis#885 over-application control positions are present."""
        combined = " ".join(source for _, source in oracle.SCORE_ONE_CONTROL_FIXTURES)
        for spelling in ("(record {}", "(kv {}", "(access {}", "(export {}", "(deftype {"):
            self.assertIn(
                spelling,
                combined,
                f"no score-one control covers the {spelling} structural position",
            )

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


class TestBoundedChild(unittest.TestCase):
    @patch("subprocess.Popen")
    def test_explicit_environment_is_forwarded_to_the_child(
        self, popen: MagicMock
    ) -> None:
        process = popen.return_value
        process.communicate.return_value = ("out", "err")
        process.returncode = 0
        child_env = {"CHELIS_ORACLE_SENTINEL": "kept"}

        completed = oracle.run_bounded_child(
            "environment handoff",
            ("chelis", "check"),
            timeout=1,
            env=child_env,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertIs(popen.call_args.kwargs["env"], child_env)

    def test_hung_child_is_reaped_and_diagnostic_names_its_obligation(self) -> None:
        command = (
            sys.executable,
            "-c",
            "import time; print('child-started', flush=True); time.sleep(60)",
        )
        with self.assertRaises(oracle.OracleFailure) as raised:
            oracle.run_bounded_child("planted hung check", command, timeout=0.05)

        message = str(raised.exception)
        self.assertIn("obligation=planted hung check", message)
        self.assertIn("state=timed_out", message)
        self.assertIn("termination=", message)
        self.assertIn("child-started", message)
        pid_match = re.search(r"pid=(\d+)", message)
        self.assertIsNotNone(pid_match, message)
        assert pid_match is not None
        with self.assertRaises(ProcessLookupError):
            os.kill(int(pid_match.group(1)), 0)

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


class TestNameInExprRejection(unittest.TestCase):
    """Test obligation 7 logic with mocked subprocess."""

    def _mock_check_error(self, source: str) -> subprocess.CompletedProcess[str]:
        """Simulate chelis check returning the [03-ROLE-2] stamp rejection."""
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
                    "message": "stamp error: bare name `oops` at expression slot; use `(var {} oops)`",
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
    def test_name_rejected_passes(self, mock_check: MagicMock) -> None:
        mock_check.side_effect = self._mock_check_error
        # Should not raise.
        oracle.check_name_in_expr_rejected()

    @patch("unrepresentable_domain_oracle.run_chelis_check")
    def test_name_accepted_fails(self, mock_check: MagicMock) -> None:
        """If chelis check returns exit 0 for a bare-name fixture, oracle fails."""
        mock_check.return_value = subprocess.CompletedProcess(
            args=["chelis", "check", "x.dp"],
            returncode=0,
            stdout='{"score": 1, "errors": []}',
            stderr="",
        )
        with self.assertRaises(oracle.OracleFailure):
            oracle.check_name_in_expr_rejected()

    @patch("unrepresentable_domain_oracle.run_chelis_check")
    def test_error_without_bare_name_identification_fails(
        self, mock_check: MagicMock
    ) -> None:
        """The diagnostic must identify the form as a bare name."""
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
            oracle.check_name_in_expr_rejected()

    @patch("unrepresentable_domain_oracle.run_chelis_check")
    def test_error_without_remediation_spelling_fails(
        self, mock_check: MagicMock
    ) -> None:
        """[03-ROLE-2]'s SHOULD is this oracle's MUST: the `(var {} ...)`
        remediation spelling has to appear, or the diagnostic regressed."""
        report = {
            "score": 0,
            "errors": [
                {
                    "kind": "Other",
                    "message": "stamp error: bare name `oops` at expression slot",
                    "severity": 0.5,
                }
            ],
        }
        mock_check.return_value = subprocess.CompletedProcess(
            args=["chelis", "check", "x.dp"],
            returncode=2,
            stdout=json.dumps(report),
            stderr="",
        )
        with self.assertRaises(oracle.OracleFailure):
            oracle.check_name_in_expr_rejected()


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
            elif identification == oracle.EMPTY_PROGRAM_IDENTIFICATION:
                # [03-PROG-3]: text that yields no top-level form at all,
                # which is exactly whitespace and `;` comment lines.
                for line in source.splitlines():
                    stripped = line.strip()
                    self.assertTrue(
                        stripped == "" or stripped.startswith(";"),
                        f"{name}: {line!r} is not whitespace or a comment",
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
            if identification == oracle.EMPTY_PROGRAM_IDENTIFICATION:
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
                oracle.check_validate_agrees_with_check,
                oracle.check_stamp_pass_integration_tests,
                oracle.check_compiler_api_ingress,
                oracle.check_name_in_expr_rejected,
                oracle.check_metadata_contract,
            ],
        )


class TestBinaryResolution(unittest.TestCase):
    """The build-once fast path must degrade to `cargo run`, never to a pass."""

    def setUp(self) -> None:
        oracle._BINARY_RESOLUTION_ATTEMPTED = False
        oracle._RESOLVED_CHELIS_BINARY = None
        # These cases are about the build path, so the handoff must be out
        # of the way. A stray CHELIS_ORACLE_BINARY in the ambient shell
        # would otherwise short-circuit resolution and make the assertions
        # below vacuously wrong (chelis#1322).
        self._env = patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop(oracle.ORACLE_BINARY_ENV, None)

    def tearDown(self) -> None:
        self._env.stop()
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


class TestHandedOverBinary(unittest.TestCase):
    """chelis#1322: an explicit handoff is authoritative and fails closed.

    `scripts/gate.py` sets `CHELIS_ORACLE_BINARY` to the `chelis` an earlier
    command in the same gate run already built. Three behaviors matter and
    are locked here: set-and-valid skips the build, set-and-invalid is a
    loud failure, and unset leaves the historical build path untouched.
    """

    def setUp(self) -> None:
        oracle._BINARY_RESOLUTION_ATTEMPTED = False
        oracle._RESOLVED_CHELIS_BINARY = None
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()
        oracle._BINARY_RESOLUTION_ATTEMPTED = False
        oracle._RESOLVED_CHELIS_BINARY = None

    def _executable(self, name: str = "chelis") -> Path:
        path = self.root / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
        return path

    def test_unset_reads_as_no_handoff(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(oracle.ORACLE_BINARY_ENV, None)
            self.assertIsNone(oracle.handed_over_binary())

    def test_empty_and_whitespace_fail_loudly_rather_than_disabling(
        self,
    ) -> None:
        # Present-but-empty is a caller who meant to hand something over,
        # not an off switch. Reading it as "unset" would let an ambient
        # `export CHELIS_ORACLE_BINARY=` silently disable the handoff, and
        # `scripts/gate.py` would not object either because its own guard
        # keys on presence. Both sides reject it instead.
        for value in ("", "   "):
            with self.subTest(value=value):
                with patch.dict(
                    os.environ, {oracle.ORACLE_BINARY_ENV: value}, clear=False
                ):
                    with self.assertRaisesRegex(
                        oracle.OracleBinaryError, "empty"
                    ):
                        oracle.handed_over_binary()

    def test_a_directory_is_named_as_a_directory(self) -> None:
        # `is_file()` alone reported an existing directory as "no file
        # exists at ...", which sends the reader looking for a missing
        # path rather than at the path they gave.
        with patch.dict(
            os.environ, {oracle.ORACLE_BINARY_ENV: str(self.root)}, clear=False
        ):
            with self.assertRaisesRegex(
                oracle.OracleBinaryError, "is a directory"
            ):
                oracle.handed_over_binary()

    def test_a_valid_absolute_path_is_returned(self) -> None:
        binary = self._executable()
        with patch.dict(
            os.environ, {oracle.ORACLE_BINARY_ENV: str(binary)}, clear=False
        ):
            self.assertEqual(oracle.handed_over_binary(), binary)

    def test_a_relative_path_resolves_against_the_repo_root(self) -> None:
        # Not against the caller's cwd: gate.py runs children from the repo
        # root, and a resolution that depended on cwd would break the moment
        # anyone invoked the oracle from a subdirectory.
        binary = self._executable("relative-chelis")
        with patch.object(oracle, "REPO_ROOT", self.root):
            with patch.dict(
                os.environ,
                {oracle.ORACLE_BINARY_ENV: "relative-chelis"},
                clear=False,
            ):
                self.assertEqual(oracle.handed_over_binary(), binary)

    def test_a_missing_path_fails_loudly(self) -> None:
        missing = self.root / "not-built" / "chelis"
        with patch.dict(
            os.environ, {oracle.ORACLE_BINARY_ENV: str(missing)}, clear=False
        ):
            with self.assertRaises(oracle.OracleBinaryError) as caught:
                oracle.handed_over_binary()
        message = str(caught.exception)
        self.assertIn(oracle.ORACLE_BINARY_ENV, message)
        self.assertIn(str(missing), message)
        self.assertIn("will not be replaced", message)

    def test_a_non_executable_path_fails_loudly(self) -> None:
        inert = self.root / "chelis.txt"
        inert.write_text("not a binary\n")
        inert.chmod(0o644)
        with patch.dict(
            os.environ, {oracle.ORACLE_BINARY_ENV: str(inert)}, clear=False
        ):
            with self.assertRaisesRegex(
                oracle.OracleBinaryError, "not executable"
            ):
                oracle.handed_over_binary()

    @patch("subprocess.run")
    def test_a_valid_handoff_skips_the_cargo_build(
        self, mock_run: MagicMock
    ) -> None:
        binary = self._executable()
        with patch.dict(
            os.environ, {oracle.ORACLE_BINARY_ENV: str(binary)}, clear=False
        ):
            self.assertEqual(oracle.resolve_chelis_binary(), binary)
        self.assertEqual(
            mock_run.call_count,
            0,
            "a handed-over binary must not re-enter cargo",
        )

    @patch("subprocess.run")
    def test_an_invalid_handoff_never_falls_back_to_a_build(
        self, mock_run: MagicMock
    ) -> None:
        # The whole point of failing closed: a silent fall back would run
        # the build this handoff exists to avoid and still report PASS.
        missing = self.root / "not-built" / "chelis"
        with patch.dict(
            os.environ, {oracle.ORACLE_BINARY_ENV: str(missing)}, clear=False
        ):
            with self.assertRaises(oracle.OracleBinaryError):
                oracle.resolve_chelis_binary()
            self.assertEqual(mock_run.call_count, 0)
            # And it stays failed: the resolution state was left untouched,
            # so a later call raises again rather than reaching cargo.
            with self.assertRaises(oracle.OracleBinaryError):
                oracle.resolve_chelis_binary()
            self.assertEqual(mock_run.call_count, 0)

    @patch("subprocess.run")
    def test_no_handoff_still_runs_the_cargo_build(
        self, mock_run: MagicMock
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["cargo", "build"], returncode=101, stdout="", stderr=""
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(oracle.ORACLE_BINARY_ENV, None)
            self.assertIsNone(oracle.resolve_chelis_binary())
        self.assertEqual(mock_run.call_count, 1)
        self.assertEqual(
            mock_run.call_args.args[0],
            ("cargo", "build", "-p", "chelis-cli", "--bin", "chelis", "--quiet"),
        )

    def test_an_invalid_handoff_is_reported_before_any_obligation_runs(
        self,
    ) -> None:
        # A harness fault is not an obligation result. `main` must abandon
        # the run without printing a per-obligation verdict, and without
        # the acceptance line.
        missing = self.root / "not-built" / "chelis"
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(
            os.environ, {oracle.ORACLE_BINARY_ENV: str(missing)}, clear=False
        ):
            with redirect_stdout(out), redirect_stderr(err):
                status = oracle.main()
        self.assertEqual(status, 2)
        self.assertIn("ORACLE: FAIL", err.getvalue())
        self.assertIn(oracle.ORACLE_BINARY_ENV, err.getvalue())
        self.assertNotIn("ORACLE: PASS", out.getvalue())
        self.assertNotIn("Obligation 1", out.getvalue())


class TestSyntheticFixtureStyleGate(unittest.TestCase):
    """Synthetic Deep fixtures must not lint their whole temp directory."""

    def setUp(self) -> None:
        oracle._BINARY_RESOLUTION_ATTEMPTED = True
        oracle._RESOLVED_CHELIS_BINARY = Path("/tmp/test-chelis")

    def tearDown(self) -> None:
        oracle._BINARY_RESOLUTION_ATTEMPTED = False
        oracle._RESOLVED_CHELIS_BINARY = None

    @patch("unrepresentable_domain_oracle.run_bounded_child")
    def test_check_disables_advisory_lint_without_dropping_the_environment(
        self, run_bounded_child: MagicMock
    ) -> None:
        run_bounded_child.return_value = subprocess.CompletedProcess(
            args=["chelis", "check"], returncode=2, stdout="{}", stderr=""
        )
        with patch.dict(os.environ, {"CHELIS_ORACLE_SENTINEL": "kept"}, clear=False):
            oracle.run_chelis_check(Path("/tmp/fixture.dp"))
        child_env = run_bounded_child.call_args.kwargs["env"]
        self.assertEqual(child_env[oracle.STYLE_GATE_DISABLE_ENV], "1")
        self.assertEqual(child_env["CHELIS_ORACLE_SENTINEL"], "kept")

    @patch("unrepresentable_domain_oracle.run_bounded_child")
    def test_validate_disables_advisory_lint_without_dropping_the_environment(
        self, run_bounded_child: MagicMock
    ) -> None:
        run_bounded_child.return_value = subprocess.CompletedProcess(
            args=["chelis", "validate"], returncode=2, stdout="{}", stderr=""
        )
        with patch.dict(os.environ, {"CHELIS_ORACLE_SENTINEL": "kept"}, clear=False):
            oracle.run_chelis_validate(Path("/tmp/fixture.dp"))
        child_env = run_bounded_child.call_args.kwargs["env"]
        self.assertEqual(child_env[oracle.STYLE_GATE_DISABLE_ENV], "1")
        self.assertEqual(child_env["CHELIS_ORACLE_SENTINEL"], "kept")


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



class TestMetadataContract(unittest.TestCase):
    def test_metadata_obligation_is_continuous_and_executes_its_suite(self) -> None:
        self.assertIn(oracle.check_metadata_contract, oracle.OBLIGATIONS)
        self.assertIn("metadata_contract", oracle.STAMP_NEXTEST_COMMAND)
        self.assertIn("metadata_ingress", oracle.STAMP_NEXTEST_COMMAND)
        self.assertIn("canonical_surf", oracle.STAMP_NEXTEST_COMMAND)
        self.assertIn("canonical_surf_roundtrip", oracle.STAMP_NEXTEST_COMMAND)

    def test_metadata_corpus_checks_both_verdicts_and_diagnostics(self) -> None:
        def check(path: Path) -> subprocess.CompletedProcess[str]:
            source = path.read_text()
            bad = next((key for _, text, key in oracle.METADATA_REJECTED_FIXTURES if text == source), None)
            report = {"score": 0.0 if bad else 1.0, "errors": [f"metadata `{bad}`"] if bad else []}
            return subprocess.CompletedProcess([], 2 if bad else 0, json.dumps(report), "")
        def validate(path: Path) -> subprocess.CompletedProcess[str]:
            result = check(path)
            return subprocess.CompletedProcess([], result.returncode, result.stdout, "")
        with patch.object(oracle, "run_chelis_check", side_effect=check), patch.object(oracle, "run_chelis_validate", side_effect=validate):
            oracle.check_metadata_contract()

    def test_metadata_rejection_cannot_be_a_false_green_or_empty_error(self) -> None:
        for report, code in [({"score": 1.0, "errors": []}, 0), ({"score": 0.0, "errors": []}, 2), ({"score": 1.0, "errors": ["surf_path"]}, 2), ({"score": 0.0, "errors": ["unrelated"]}, 2)]:
            result = subprocess.CompletedProcess([], code, json.dumps(report), "")
            with self.subTest(report=report), patch.object(oracle, "run_chelis_check", return_value=result), patch.object(oracle, "run_chelis_validate", return_value=result):
                with self.assertRaises(oracle.OracleFailure):
                    oracle.check_metadata_contract()

    def test_metadata_grammar_leg_cannot_accept_a_rejected_fixture(self) -> None:
        check = subprocess.CompletedProcess([], 2, json.dumps({"score": 0, "errors": ["surf_path"]}), "")
        validate = subprocess.CompletedProcess([], 0, "", "")
        with patch.object(oracle, "run_chelis_check", return_value=check), patch.object(oracle, "run_chelis_validate", return_value=validate):
            with self.assertRaises(oracle.OracleFailure):
                oracle.check_metadata_contract()

if __name__ == "__main__":
    unittest.main()
