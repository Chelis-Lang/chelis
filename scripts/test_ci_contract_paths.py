from __future__ import annotations

import io
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

from scripts import ci_contract_paths as contract


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_WORKFLOWS = (
    "ci.yml",
    "conformance.yml",
    "changelog.yml",
    "pr-contract-acknowledgements.yml",
    "pr-base-retarget.yml",
    "secret-scan.yml",
)
SCRIPT_PATH = re.compile(r"(?:\.github/)?scripts/[A-Za-z0-9_./-]+\.py")
SCRIPT_MODULE = re.compile(r"scripts\.[A-Za-z0-9_]+")


class CiContractPathTests(unittest.TestCase):
    def test_required_validation_controllers_are_contract_paths(self) -> None:
        for path in (
            ".github/workflows/ci.yml",
            ".github/actions/free-disk-space/action.yml",
            ".config/ci-change-owned-durations.json",
            ".config/ci-test-targets.toml",
            "AGENTS.md",
            "scripts/ci_candidate_receipt.py",
            "scripts/test_ci_candidate_receipt.py",
            "scripts/gate.py",
            "scripts/test_gate.py",
            "scripts/test_gate_local.py",
            "scripts/changelog.py",
            "scripts/phase3_test_change_report.py",
            "scripts/phase4b_change_report.py",
            "scripts/check_agent_skills.py",
            "scripts/dtype_phase4b_oracle.py",
        ):
            with self.subTest(path=path):
                self.assertTrue(contract.is_ci_contract_path(path))

    def test_unrelated_implementation_paths_remain_eligible(self) -> None:
        paths = ["README.md", "crates/chelis-ir/src/lower.rs"]
        self.assertFalse(contract.ci_contract_changed(paths))
        self.assertEqual(contract.reuse_eligibility(paths), (True, []))

    def test_every_required_workflow_controller_is_a_contract_path(self) -> None:
        controllers: set[str] = set()
        for workflow in REQUIRED_WORKFLOWS:
            text = (ROOT / ".github/workflows" / workflow).read_text()
            controllers.update(SCRIPT_PATH.findall(text))
            controllers.update(
                module.replace(".", "/", 1) + ".py"
                for module in SCRIPT_MODULE.findall(text)
            )
        missing = sorted(
            path
            for path in controllers
            if not contract.is_ci_contract_path(path)
        )
        self.assertEqual(missing, [])

    def test_empty_change_set_fails_safe(self) -> None:
        self.assertTrue(contract.ci_contract_changed([]))

    def test_reuse_blockers_use_the_same_classifier(self) -> None:
        paths = [
            "crates/chelis-ir/src/lower.rs",
            "scripts/changelog.py",
            "scripts/test_gate.py",
            ".github/workflows/ci.yml",
        ]
        self.assertEqual(
            contract.reuse_eligibility(paths),
            (
                False,
                [
                    ".github/workflows/ci.yml",
                    "scripts/changelog.py",
                    "scripts/test_gate.py",
                ],
            ),
        )

    def test_cli_emits_the_github_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "github-output")
            original_stdin = sys.stdin
            original_output = os.environ.get("GITHUB_OUTPUT")
            sys.stdin = io.StringIO("scripts/changelog.py\n")
            os.environ["GITHUB_OUTPUT"] = output
            try:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    self.assertEqual(contract.main(), 0)
                self.assertEqual(
                    stdout.getvalue(), "ci_contract_changed=true\n"
                )
                with open(output, encoding="utf-8") as handle:
                    self.assertEqual(
                        handle.read(), "ci_contract_changed=true\n"
                    )
            finally:
                sys.stdin = original_stdin
                if original_output is None:
                    os.environ.pop("GITHUB_OUTPUT", None)
                else:
                    os.environ["GITHUB_OUTPUT"] = original_output


if __name__ == "__main__":
    unittest.main()
