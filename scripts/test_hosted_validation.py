"""Keep the CI coverage that replaces mandatory --local execution live."""

import copy
from pathlib import Path
import unittest

import yaml

from scripts import gate


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "uv run --managed-python --python 3.11 --no-project --with PyYAML==6.0.3 python"
SKILL_COMMANDS = (
    f"{PYTHON} scripts/check_agent_skills.py",
    f"{PYTHON} -m unittest scripts.test_check_agent_skills scripts.test_hosted_validation",
    "cargo test -p chelis-conformance --test asset_drift_tripwire --test skill_set_uniformity",
)


def assert_hosted_coverage(test, workflow):
    jobs = workflow["jobs"]
    docs = jobs["docs"]
    test.assertNotIn("if", docs)
    test.assertFalse(docs.get("continue-on-error", False))
    for command in SKILL_COMMANDS:
        steps = [step for step in docs["steps"] if step.get("run") == command]
        test.assertEqual(len(steps), 1, command)
        test.assertNotIn("if", steps[0])
        test.assertFalse(steps[0].get("continue-on-error", False))
    macos = jobs["macos-workspace-shard"]
    test.assertEqual(macos["runs-on"], "macos-latest")
    test.assertEqual(macos.get("if"), "${{ !cancelled() && (needs.changes.result != 'success' || needs.changes.outputs.docs_only != 'true') }}")
    test.assertFalse(macos.get("continue-on-error", False))
    test.assertIn(1, macos["strategy"]["matrix"]["shard"])
    toolchains = [step for step in macos["steps"] if step.get("uses", "").startswith("dtolnay/rust-toolchain@")]
    test.assertEqual(len(toolchains), 1)
    test.assertIn("clippy", toolchains[0].get("with", {}).get("components", "").split(","))
    for command in (gate.CLIPPY_WORKSPACE, gate.CLIPPY_SOLVER_FREE_FEATURES):
        steps = [step for step in macos["steps"] if step.get("run") == " ".join(command)]
        test.assertEqual(len(steps), 1, command)
        test.assertEqual(steps[0].get("if"), "matrix.shard == 1")
        test.assertFalse(steps[0].get("continue-on-error", False))
    aggregate = jobs["macos-smoke"]
    test.assertIn("macos-workspace-shard", aggregate["needs"])
    test.assertEqual(aggregate.get("if"), "${{ always() && (needs.changes.result != 'success' || needs.changes.outputs.docs_only != 'true') }}")
    test.assertFalse(aggregate.get("continue-on-error", False))
    command = "python3 scripts/ci_require_success.py macos-workspace-shard=${{ needs.macos-workspace-shard.result }}"
    steps = [step for step in aggregate["steps"] if step.get("run") == command]
    test.assertEqual(len(steps), 1)
    test.assertNotIn("if", steps[0])
    test.assertFalse(steps[0].get("continue-on-error", False))


class HostedCoverageTests(unittest.TestCase):
    def setUp(self):
        self.workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())

    def test_current_workflow_preserves_hosted_coverage(self):
        assert_hosted_coverage(self, self.workflow)

    def test_missing_or_nonblocking_skill_check_is_rejected(self):
        for command in SKILL_COMMANDS:
            for change in ("remove", "skip", "ignore-failure"):
                with self.subTest(command=command, change=change):
                    workflow = copy.deepcopy(self.workflow)
                    steps = workflow["jobs"]["docs"]["steps"]
                    step = next(step for step in steps if step.get("run") == command)
                    if change == "remove":
                        steps.remove(step)
                    elif change == "skip":
                        step["if"] = "false"
                    else:
                        step["continue-on-error"] = True
                    with self.assertRaises(AssertionError):
                        assert_hosted_coverage(self, workflow)

    def test_docs_checks_cannot_be_skipped_or_made_nonblocking(self):
        for key, value in (("if", "false"), ("continue-on-error", True)):
            with self.subTest(key=key):
                workflow = copy.deepcopy(self.workflow)
                workflow["jobs"]["docs"][key] = value
                with self.assertRaises(AssertionError):
                    assert_hosted_coverage(self, workflow)

    def test_clippy_must_run_on_macos_in_both_configurations(self):
        for change in ("linux", "no-shard-one", "missing", "wrong-shard", "ignore-failure"):
            with self.subTest(change=change):
                workflow = copy.deepcopy(self.workflow)
                job = workflow["jobs"]["macos-workspace-shard"]
                if change == "linux":
                    job["runs-on"] = "ubuntu-latest"
                elif change == "no-shard-one":
                    job["strategy"]["matrix"]["shard"] = [2]
                else:
                    step = next(step for step in job["steps"] if step.get("run") == " ".join(gate.CLIPPY_SOLVER_FREE_FEATURES))
                    if change == "missing":
                        job["steps"].remove(step)
                    elif change == "wrong-shard":
                        step["if"] = "matrix.shard == 3"
                    else:
                        step["continue-on-error"] = True
                with self.assertRaises(AssertionError):
                    assert_hosted_coverage(self, workflow)

    def test_macos_producer_and_aggregate_cannot_be_skipped_or_made_nonblocking(self):
        for job_name in ("macos-workspace-shard", "macos-smoke"):
            for key, value in (("if", "false"), ("continue-on-error", True)):
                with self.subTest(job=job_name, key=key):
                    workflow = copy.deepcopy(self.workflow)
                    workflow["jobs"][job_name][key] = value
                    with self.assertRaises(AssertionError):
                        assert_hosted_coverage(self, workflow)

    def test_macos_aggregate_must_enforce_the_shard_result(self):
        for change in ("remove", "skip", "ignore-failure", "wrong-result"):
            with self.subTest(change=change):
                workflow = copy.deepcopy(self.workflow)
                steps = workflow["jobs"]["macos-smoke"]["steps"]
                step = next(step for step in steps if "ci_require_success.py" in step.get("run", ""))
                if change == "remove":
                    steps.remove(step)
                elif change == "skip":
                    step["if"] = "false"
                elif change == "ignore-failure":
                    step["continue-on-error"] = True
                else:
                    step["run"] = step["run"].replace("needs.macos-workspace-shard.result", "needs.changes.result")
                with self.assertRaises(AssertionError):
                    assert_hosted_coverage(self, workflow)


if __name__ == "__main__":
    unittest.main()
