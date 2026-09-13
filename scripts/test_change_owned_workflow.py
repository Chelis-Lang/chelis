"""Lock the required change-owned lane and isolated package-expansion trial."""

from __future__ import annotations

import copy
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/ci.yml"
SHARDS = [0, 1, 2, 3]


def _run_steps(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def assert_change_owned_topology(test: unittest.TestCase, workflow: dict) -> None:
    jobs = workflow["jobs"]
    expected = {
        "integration-plan",
        "change-owned-shard",
        "change-owned-report",
        "package-expansion-shard",
        "package-expansion-summary",
    }
    test.assertLessEqual(expected, set(jobs))
    for name in expected:
        steps = jobs[name]["steps"]
        invoke = next(index for index, step in enumerate(steps)
                      if "scripts/ci_change_owned.py" in step.get("run", ""))
        setup = next((index for index, step in enumerate(steps)
                      if step.get("run") in {"uv venv --python 3.11", "python3 scripts/ci_setup_uv_python.py"}), None)
        test.assertIsNotNone(setup, f"{name} requires managed Python setup")
        test.assertLess(setup, invoke)
        test.assertIn(".venv/bin/python scripts/ci_change_owned.py", steps[invoke]["run"])

    planner = jobs["integration-plan"]
    test.assertEqual(planner["needs"], ["changes"])
    test.assertEqual(planner["timeout-minutes"], 10)
    test.assertFalse(planner.get("continue-on-error", False))
    test.assertIn("needs.changes.outputs.docs_only != 'true'", planner["if"])
    test.assertIn("needs.changes.result != 'success'", planner["if"])
    checkout = next(
        step
        for step in planner["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )
    test.assertEqual(checkout.get("with", {}).get("fetch-depth"), 0)
    test.assertIn("scripts/ci_change_owned.py plan", _run_steps(planner))
    test.assertIn("integration-change-plan", str(planner))
    test.assertIn("target/integration-change/plan.json", str(planner))

    required = jobs["change-owned-shard"]
    test.assertEqual(required["needs"], ["changes", "integration-plan"])
    test.assertEqual(required["timeout-minutes"], 20)
    test.assertFalse(required.get("continue-on-error", False))
    test.assertFalse(required["strategy"]["fail-fast"])
    test.assertEqual(required["strategy"]["matrix"]["shard"], SHARDS)
    test.assertIn("needs.integration-plan.result == 'success'", required["if"])
    test.assertIn(
        "scripts/ci_change_owned.py run-shard", _run_steps(required)
    )
    test.assertIn("--lane change-owned", _run_steps(required))
    test.assertIn(
        "integration-change-owned-${{ matrix.shard }}", str(required)
    )
    required_upload = next(
        step
        for step in required["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    test.assertEqual(
        required_upload.get("with", {}).get("if-no-files-found"), "error"
    )

    required_report = jobs["change-owned-report"]
    test.assertEqual(
        required_report["needs"],
        ["changes", "integration-plan", "change-owned-shard"],
    )
    test.assertIn("always()", required_report["if"])
    test.assertFalse(required_report.get("continue-on-error", False))
    required_report_commands = _run_steps(required_report)
    test.assertIn("scripts/ci_require_success.py", required_report_commands)
    test.assertIn(
        "change-owned-shard=${{ needs.change-owned-shard.result }}",
        required_report_commands,
    )
    test.assertIn("scripts/ci_change_owned.py report", required_report_commands)
    test.assertIn("--lane change-owned", required_report_commands)
    test.assertIn("--required", required_report_commands)

    stable = jobs["integration"]
    test.assertEqual(
        stable["needs"], ["changes", "ci-fast", "change-owned-report"]
    )
    stable_commands = _run_steps(stable)
    test.assertIn("ci-fast=${{ needs.ci-fast.result }}", stable_commands)
    test.assertIn(
        "change-owned-report=${{ needs.change-owned-report.result }}",
        stable_commands,
    )
    test.assertNotIn("package-expansion", str(stable))

    expansion = jobs["package-expansion-shard"]
    test.assertEqual(
        expansion["needs"],
        ["changes", "integration-plan", "change-owned-report"],
    )
    test.assertEqual(expansion["timeout-minutes"], 20)
    test.assertFalse(expansion.get("continue-on-error", False))
    test.assertFalse(expansion["strategy"]["fail-fast"])
    test.assertEqual(expansion["strategy"]["matrix"]["shard"], SHARDS)
    test.assertIn(
        "needs.change-owned-report.result == 'success'", expansion["if"]
    )
    expansion_commands = _run_steps(expansion)
    test.assertIn("scripts/ci_change_owned.py run-shard", expansion_commands)
    test.assertIn("--lane package-expansion", expansion_commands)
    test.assertIn(
        "integration-package-expansion-${{ matrix.shard }}", str(expansion)
    )

    summary = jobs["package-expansion-summary"]
    test.assertEqual(
        summary["needs"],
        [
            "changes",
            "integration-plan",
            "change-owned-report",
            "package-expansion-shard",
        ],
    )
    test.assertIn("always()", summary["if"])
    test.assertFalse(summary.get("continue-on-error", False))
    summary_commands = _run_steps(summary)
    test.assertIn("scripts/ci_change_owned.py report", summary_commands)
    test.assertIn("--lane package-expansion", summary_commands)
    test.assertNotIn("--required", summary_commands)


class ChangeOwnedWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = yaml.safe_load(WORKFLOW.read_text())

    def test_required_lane_and_informational_trial_are_isolated(self) -> None:
        assert_change_owned_topology(self, self.workflow)

    def test_missing_managed_python_setup_or_system_invocation_is_rejected(self) -> None:
        for mutation in ("missing-setup", "system-python"):
            workflow = copy.deepcopy(self.workflow)
            planner = workflow["jobs"]["integration-plan"]
            if mutation == "missing-setup":
                planner["steps"] = [step for step in planner["steps"]
                                    if step.get("run") != "uv venv --python 3.11"]
            else:
                for step in planner["steps"]:
                    if "scripts/ci_change_owned.py" in step.get("run", ""):
                        step["run"] = step["run"].replace(".venv/bin/python", "python3")
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                assert_change_owned_topology(self, workflow)

    def test_expansion_cannot_feed_or_run_ahead_of_the_required_verdict(self) -> None:
        for mutation in ("feeds-stable", "runs-concurrently"):
            workflow = copy.deepcopy(self.workflow)
            if mutation == "feeds-stable":
                workflow["jobs"]["integration"]["needs"].append(
                    "package-expansion-summary"
                )
            else:
                workflow["jobs"]["package-expansion-shard"]["needs"].remove(
                    "change-owned-report"
                )
            with self.subTest(mutation=mutation), self.assertRaises(
                AssertionError
            ):
                assert_change_owned_topology(self, workflow)

    def test_required_report_cannot_be_downgraded(self) -> None:
        for mutation in ("skip-cancellation", "ignore-failure", "drop-worker"):
            workflow = copy.deepcopy(self.workflow)
            report = workflow["jobs"]["change-owned-report"]
            if mutation == "skip-cancellation":
                report["if"] = report["if"].replace("always()", "!cancelled()")
            elif mutation == "ignore-failure":
                report["continue-on-error"] = True
            else:
                report["needs"].remove("change-owned-shard")
            with self.subTest(mutation=mutation), self.assertRaises(
                AssertionError
            ):
                assert_change_owned_topology(self, workflow)


if __name__ == "__main__":
    unittest.main()
