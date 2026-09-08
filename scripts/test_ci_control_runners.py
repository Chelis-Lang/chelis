"""Short control jobs can use an isolated pool without changing their verdicts."""

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTROL_RUNNER = "${{ vars.CHELIS_CONTROL_RUNNER || 'ubuntu-latest' }}"
CONTROL_JOBS = {
    "ci.yml": {"changes", "rejection-authority-liveness", "lint-and-unit",
               "workspace-tests", "generalize-sweep-oracle", "integration",
               "macos-smoke", "test-telemetry", "no-ai-authorship"},
    "conformance.yml": {"changes"},
    "changelog.yml": {"changelog"},
}


def assert_routing(test, workflow, selected):
    test.assertTrue(selected <= workflow["jobs"].keys())
    for name, job in workflow["jobs"].items():
        if name in selected:
            test.assertEqual(job["runs-on"], CONTROL_RUNNER, name)
        else:
            test.assertNotIn("CHELIS_CONTROL_RUNNER", str(job.get("runs-on")), name)


class ControlRunnerTests(unittest.TestCase):
    def test_only_short_jobs_use_control_capacity(self):
        for file, selected in CONTROL_JOBS.items():
            with self.subTest(file=file):
                workflow = yaml.safe_load((ROOT / ".github/workflows" / file).read_text())
                assert_routing(self, workflow, selected)

    def test_build_job_cannot_take_control_capacity(self):
        workflow = {"jobs": {"build": {"runs-on": CONTROL_RUNNER}}}
        with self.assertRaises(AssertionError):
            assert_routing(self, workflow, set())

    def test_control_job_cannot_silently_return_to_shared_capacity(self):
        workflow = {"jobs": {"aggregate": {"runs-on": "ubuntu-latest"}}}
        with self.assertRaises(AssertionError):
            assert_routing(self, workflow, {"aggregate"})

    def test_selected_job_cannot_disappear(self):
        with self.assertRaises(AssertionError):
            assert_routing(self, {"jobs": {}}, {"aggregate"})


if __name__ == "__main__":
    unittest.main()
