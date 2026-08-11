from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_openspec.py"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "openspec-validate.yml"
CI_ACTION_SHA = "9d4d4c59e46a0672b5e17a5644e99700821c87e6"
CHECKOUT_SHA = "3d3c42e5aac5ba805825da76410c181273ba90b1"
REMOTE_USES = re.compile(r"(?m)^\s*uses:\s*([^@\s]+)@([^\s#]+)")
MODE_VALUE = re.compile(r"(?m)^\s+mode:\s*(\S+)\s*$")

SPEC = importlib.util.spec_from_file_location("check_openspec", CHECKER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load OpenSpec checker: {CHECKER_PATH}")
check_openspec = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_openspec
SPEC.loader.exec_module(check_openspec)


def workflow_diagnostics(text: str) -> list[str]:
    diagnostics: list[str] = []
    expected_references = [
        ("actions/checkout", CHECKOUT_SHA),
        ("Chelis-Lang/ci/actions/setup-devenv", CI_ACTION_SHA),
        ("Chelis-Lang/ci/actions/openspec-governance", CI_ACTION_SHA),
    ]
    if REMOTE_USES.findall(text) != expected_references:
        diagnostics.append("action-provenance")
    if text.count("fetch-depth: 0") != 1:
        diagnostics.append("full-history")
    if text.count("persist-credentials: false") != 1:
        diagnostics.append("credentials")
    if MODE_VALUE.findall(text) != ["advisory"]:
        diagnostics.append("advisory-mode")
    for forbidden in ("npm install", "npm ci", "openspec validate", "setup-node"):
        if forbidden in text:
            diagnostics.append(f"local-provisioning:{forbidden}")
    return diagnostics


class WorkflowContractTests(unittest.TestCase):
    def test_workflow_uses_latest_central_advisory_action(self) -> None:
        text = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertEqual(workflow_diagnostics(text), [])

    def test_workflow_contract_rejects_negative_mutations(self) -> None:
        text = WORKFLOW_PATH.read_text(encoding="utf-8")
        fixtures = {
            "mutable-action": text.replace(CI_ACTION_SHA, "main", 1),
            "missing-setup": text.replace(
                "      - name: Set up portable Devenv\n"
                f"        uses: Chelis-Lang/ci/actions/setup-devenv@{CI_ACTION_SHA} "
                "# ci#54 colored selective retry\n\n",
                "",
            ),
            "mutable-checkout": text.replace(CHECKOUT_SHA, "v7"),
            "enforced-mode": text.replace("mode: advisory", "mode: enforce"),
            "second-mode": text + "\n# fixture\n          mode: enforce\n",
            "second-action": (
                text
                + "\n# fixture\n"
                + "        uses: Chelis-Lang/ci/actions/openspec-governance@main\n"
            ),
            "shallow-history": text.replace("          fetch-depth: 0\n", ""),
            "local-install": text + "\n# npm install @fission-ai/openspec\n",
        }
        for name, fixture in fixtures.items():
            with self.subTest(name=name):
                self.assertTrue(workflow_diagnostics(fixture))


class CheckerContractTests(unittest.TestCase):
    def test_fixed_action_invocation_parses_to_a_precise_type(self) -> None:
        invocation = check_openspec.parse_invocation(
            ["--self-test", "--merge-bound", "--base", "a" * 40],
            {"OPENSPEC_BIN": "/action/node_modules/.bin/openspec"},
        )
        self.assertEqual(str(invocation.base), "a" * 40)
        self.assertEqual(
            invocation.openspec,
            Path("/action/node_modules/.bin/openspec"),
        )
        self.assertEqual(
            check_openspec.build_command(invocation),
            (
                "/action/node_modules/.bin/openspec",
                "validate",
                "--all",
                "--strict",
                "--no-interactive",
            ),
        )

    def test_invalid_action_invocations_fail_at_the_boundary(self) -> None:
        fixtures = (
            ([], {"OPENSPEC_BIN": "/bin/openspec"}),
            (
                ["--self-test", "--merge-bound", "--base", "main"],
                {"OPENSPEC_BIN": "/bin/openspec"},
            ),
            (
                ["--merge-bound", "--self-test", "--base", "a" * 40],
                {"OPENSPEC_BIN": "/bin/openspec"},
            ),
            (
                ["--self-test", "--merge-bound", "--base", "origin/main"],
                {},
            ),
        )
        for arguments, environment in fixtures:
            with self.subTest(arguments=arguments):
                with self.assertRaises(check_openspec.CheckError):
                    check_openspec.parse_invocation(arguments, environment)

    def test_validation_results_preserve_the_action_exit_contract(self) -> None:
        invocation = check_openspec.parse_invocation(
            ["--self-test", "--merge-bound", "--base", "origin/main"],
            {"OPENSPEC_BIN": "/bin/openspec"},
        )

        def runner_with(returncode: int):
            def run(*_args: object, **_kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(returncode=returncode)

            return run

        self.assertEqual(check_openspec.run_validation(invocation, runner_with(0)), 0)
        self.assertEqual(check_openspec.run_validation(invocation, runner_with(1)), 1)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            for returncode in (2, 17, -9):
                with self.subTest(returncode=returncode):
                    self.assertEqual(
                        check_openspec.run_validation(
                            invocation, runner_with(returncode)
                        ),
                        2,
                    )
        self.assertIn("operational exit code 17", stderr.getvalue())

    def test_spawn_and_timeout_failures_are_operational(self) -> None:
        invocation = check_openspec.parse_invocation(
            ["--self-test", "--merge-bound", "--base", "origin/main"],
            {"OPENSPEC_BIN": "/bin/openspec"},
        )
        failures = (OSError("missing"), subprocess.TimeoutExpired("openspec", 300))
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            for failure in failures:
                with self.subTest(failure=type(failure).__name__):
                    runner = mock.Mock(side_effect=failure)
                    self.assertEqual(
                        check_openspec.run_validation(invocation, runner), 2
                    )
        self.assertIn("OpenSpec execution failed", stderr.getvalue())

    def test_boundary_error_returns_operational_status(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = check_openspec.main([], {})
        self.assertEqual(status, 2)
        self.assertIn("check_openspec:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
