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
METADATA_ONLY = (
    "github.event_name == 'pull_request' && "
    "github.event.action == 'edited' && "
    "github.event.changes.base == null"
)


def assert_hosted_coverage(test, workflow, nightly):
    jobs = workflow["jobs"]
    docs = jobs["docs"]
    test.assertEqual(docs.get("needs"), ["changes"])
    test.assertIn(METADATA_ONLY, docs.get("if", ""))
    test.assertNotIn("docs_only", docs.get("if", ""))
    test.assertFalse(docs.get("continue-on-error", False))
    for command in SKILL_COMMANDS:
        steps = [step for step in docs["steps"] if step.get("run") == command]
        test.assertEqual(len(steps), 1, command)
        test.assertNotIn("if", steps[0])
        test.assertFalse(steps[0].get("continue-on-error", False))
    test.assertNotIn("macos-workspace-shard", jobs)
    test.assertNotIn("smt-build-darwin-arm64", jobs)
    triggers = nightly[True]  # PyYAML YAML 1.1 reads the Actions key `on` as True.
    test.assertEqual(set(triggers), {"schedule", "workflow_dispatch"})
    test.assertEqual(triggers["schedule"], [{"cron": "17 4 * * *"}])
    jobs = nightly["jobs"]
    test.assertEqual(jobs["smt-build-darwin-arm64"]["timeout-minutes"], 60)
    test.assertNotIn("if", jobs["smt-build-darwin-arm64"])
    macos = jobs["macos-workspace-shard"]
    test.assertEqual(macos["runs-on"], "macos-latest")
    test.assertNotIn("if", macos)
    test.assertEqual(macos["timeout-minutes"], 45)
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
    test.assertEqual(aggregate.get("if"), "always()")
    test.assertFalse(aggregate.get("continue-on-error", False))
    command = "python3 scripts/ci_require_success.py macos-workspace-shard=${{ needs.macos-workspace-shard.result }}"
    steps = [step for step in aggregate["steps"] if step.get("run") == command]
    test.assertEqual(len(steps), 1)
    test.assertNotIn("if", steps[0])
    test.assertFalse(steps[0].get("continue-on-error", False))


class HostedCoverageTests(unittest.TestCase):
    def setUp(self):
        self.workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
        self.nightly = yaml.safe_load((ROOT / ".github/workflows/macos-nightly.yml").read_text())

    def test_current_workflow_preserves_hosted_coverage(self):
        assert_hosted_coverage(self, self.workflow, self.nightly)

    def test_ordinary_events_cannot_trigger_macos_nightly(self):
        for event in ("push", "pull_request", "pull_request_target", "workflow_run"):
            with self.subTest(event=event):
                nightly = copy.deepcopy(self.nightly)
                nightly[True][event] = None
                with self.assertRaises(AssertionError):
                    assert_hosted_coverage(self, self.workflow, nightly)

    def test_darwin_asset_producer_requires_nightly_or_manual_event(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/build-cvc5.yml").read_text())
        self.assertEqual(workflow[True]["schedule"], [{"cron": "17 1 * * *"}])
        self.assertEqual(
            workflow["jobs"]["build-darwin-arm64"]["if"],
            "(github.event_name == 'schedule' || github.event_name == 'workflow_dispatch') && needs.plan.outputs.missing_darwin_arm64 == 'true'",
        )

    def test_ordinary_workflows_cannot_allocate_macos_runners(self):
        for path in (ROOT / ".github/workflows").glob("*.yml"):
            workflow = yaml.safe_load(path.read_text())
            events = workflow.get(True, {})
            if not {"push", "pull_request", "pull_request_target"}.intersection(events):
                continue
            if path.name == "release.yml":
                self.assertEqual(events, {"push": {"tags": ["v*"]}, "workflow_dispatch": None})
                self.assertEqual(workflow["jobs"]["build-darwin-arm64"]["runs-on"], "macos-latest")
                continue
            for name, job in workflow.get("jobs", {}).items():
                if "macos" not in str(job.get("runs-on", "")):
                    continue
                self.assertEqual((path.name, name), ("build-cvc5.yml", "build-darwin-arm64"))
                self.assertIn("github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'", job["if"])

    def test_pr_telemetry_has_no_nightly_dependencies_or_downloads(self):
        job = self.workflow["jobs"]["test-telemetry"]
        self.assertNotIn("macos", str(job))

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
                        assert_hosted_coverage(self, workflow, self.nightly)

    def test_docs_checks_cannot_be_skipped_or_made_nonblocking(self):
        for key, value in (
            ("if", "false"),
            ("if", "${{ !cancelled() }}"),
            (
                "if",
                "${{ !cancelled() && needs.changes.outputs.docs_only != 'true' }}",
            ),
            ("needs", []),
            ("continue-on-error", True),
        ):
            with self.subTest(key=key, value=value):
                workflow = copy.deepcopy(self.workflow)
                workflow["jobs"]["docs"][key] = value
                with self.assertRaises(AssertionError):
                    assert_hosted_coverage(self, workflow, self.nightly)

    def test_clippy_must_run_on_macos_in_both_configurations(self):
        for change in ("linux", "no-shard-one", "missing", "wrong-shard", "ignore-failure"):
            with self.subTest(change=change):
                workflow = copy.deepcopy(self.nightly)
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
                    assert_hosted_coverage(self, self.workflow, workflow)

    def test_macos_producer_and_aggregate_cannot_be_skipped_or_made_nonblocking(self):
        for job_name in ("macos-workspace-shard", "macos-smoke"):
            for key, value in (("if", "false"), ("continue-on-error", True)):
                with self.subTest(job=job_name, key=key):
                    workflow = copy.deepcopy(self.nightly)
                    workflow["jobs"][job_name][key] = value
                    with self.assertRaises(AssertionError):
                        assert_hosted_coverage(self, self.workflow, workflow)

    def test_macos_aggregate_must_enforce_the_shard_result(self):
        for change in ("remove", "skip", "ignore-failure", "wrong-result"):
            with self.subTest(change=change):
                workflow = copy.deepcopy(self.nightly)
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
                    assert_hosted_coverage(self, self.workflow, workflow)


if __name__ == "__main__":
    unittest.main()
