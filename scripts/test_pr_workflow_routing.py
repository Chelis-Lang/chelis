"""Lock metadata-only CI routing and the manual expansion interface."""

from __future__ import annotations

import copy
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github/workflows/ci.yml"
ACKNOWLEDGEMENTS = ROOT / ".github/workflows/pr-contract-acknowledgements.yml"
EXPANSION = ROOT / ".github/workflows/pr-package-expansion.yml"
AGENTS = ROOT / "AGENTS.md"
AUTHOR_GUIDE = ROOT / "docs/guard_changes_for_pr_authors.md"
REQUIRED_IMPLEMENTATION_JOBS = {
    "lint-and-unit": "Lint and Unit Tests (Linux)",
    "integration": "Integration Tests (Linux)",
    "backend-sanitizers": "Backend Sanitizers",
    "smt-build": "SMT Feature Build (Linux)",
    "docs": "Docs",
    "no-ai-authorship": "No AI authorship markers",
}
METADATA_ONLY = (
    "github.event_name == 'pull_request'"
    " && github.event.action == 'edited'"
    " && github.event.changes.base == null"
)


def actions_events(workflow: dict) -> dict:
    return workflow.get("on", workflow.get(True))


def assert_ci_metadata_routing(test: unittest.TestCase, workflow: dict) -> None:
    events = actions_events(workflow)
    test.assertEqual(
        set(events["pull_request"]["types"]),
        {"opened", "synchronize", "reopened", "edited"},
    )
    concurrency = str(workflow["concurrency"])
    test.assertIn("github.event.changes.base", concurrency)
    test.assertIn("metadata", concurrency)
    test.assertIn("implementation", concurrency)

    changes = workflow["jobs"]["changes"]
    test.assertIn("implementation_required", changes["outputs"])
    policy = next(
        step for step in changes["steps"] if step.get("id") == "event-policy"
    )
    test.assertIn("github.event.changes.base", str(policy))
    test.assertIn("implementation_required=false", policy["run"])
    checkout = next(
        step
        for step in changes["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )
    test.assertIn("steps.event-policy.outputs.implementation_required", checkout["if"])

    for job_id, required_name in REQUIRED_IMPLEMENTATION_JOBS.items():
        job = workflow["jobs"][job_id]
        test.assertIn("changes", job["needs"])
        test.assertIn(METADATA_ONLY, job["if"])
        test.assertIn(METADATA_ONLY, job["name"])
        test.assertIn(required_name, job["name"])
        test.assertIn("metadata-only edit", job["name"])

    for job_id, job in workflow["jobs"].items():
        if job_id == "changes":
            continue
        test.assertIn(
            METADATA_ONLY,
            job.get("if", ""),
            f"{job_id} must not execute implementation work for body/title edits",
        )

    test.assertNotIn("package-expansion-shard", workflow["jobs"])
    test.assertNotIn("package-expansion-summary", workflow["jobs"])


def assert_acknowledgement_workflow(test: unittest.TestCase, workflow: dict) -> None:
    events = actions_events(workflow)
    test.assertEqual(
        set(events["pull_request"]["types"]),
        {"opened", "synchronize", "reopened", "edited"},
    )
    test.assertIn("pull_request.number", str(workflow["concurrency"]))
    job = workflow["jobs"]["acknowledgements"]
    test.assertEqual(job["name"], "PR Contract Acknowledgements")
    test.assertEqual(job["permissions"], {"contents": "read"})
    text = str(job)
    test.assertNotIn("cargo ", text)
    test.assertNotIn("rust-toolchain", text)
    test.assertIn("fetch-depth", text)
    test.assertIn("phase3_test_change_report.py", text)
    test.assertIn("phase4b_change_report.py", text)
    test.assertEqual(text.count("--require-acknowledgement"), 2)
    test.assertIn("github.event.pull_request.head.sha", text)
    test.assertIn("github.event.pull_request.body", text)
    for artifact in ("phase3-test-changes", "phase4b-contract-changes"):
        test.assertIn(artifact, text)


def assert_manual_expansion_workflow(test: unittest.TestCase, workflow: dict) -> None:
    events = actions_events(workflow)
    test.assertEqual(set(events), {"workflow_dispatch"})
    inputs = events["workflow_dispatch"]["inputs"]
    test.assertEqual(set(inputs), {"pr_number", "expected_head_sha"})
    test.assertEqual(inputs["pr_number"]["type"], "number")
    test.assertTrue(inputs["pr_number"]["required"])
    test.assertEqual(inputs["expected_head_sha"]["type"], "string")
    test.assertTrue(inputs["expected_head_sha"]["required"])
    test.assertIn("inputs.pr_number", str(workflow["concurrency"]))

    jobs = workflow["jobs"]
    test.assertEqual(
        set(jobs),
        {"integration-plan", "package-expansion-shard", "package-expansion-summary"},
    )
    planner = jobs["integration-plan"]
    planner_text = str(planner)
    test.assertIn("pull-requests", str(planner["permissions"]))
    test.assertIn("refs/pull/", planner_text)
    test.assertIn("expected_head_sha", planner_text)
    test.assertIn("state", planner_text)
    test.assertIn("--event-name", planner_text)
    test.assertIn("pull_request", planner_text)
    test.assertIn("--pr-head", planner_text)

    worker = jobs["package-expansion-shard"]
    test.assertEqual(worker["needs"], ["integration-plan"])
    test.assertEqual(worker["strategy"]["matrix"]["shard"], [0, 1, 2, 3])
    test.assertIn("--lane package-expansion", str(worker))
    summary = jobs["package-expansion-summary"]
    test.assertEqual(
        summary["needs"], ["integration-plan", "package-expansion-shard"]
    )
    test.assertIn("always()", summary["if"])
    summary_text = str(summary)
    test.assertIn("--lane package-expansion", summary_text)
    test.assertNotIn("--required", str(summary))
    test.assertIn(".head.sha", summary_text)
    test.assertIn(".base.sha", summary_text)
    test.assertIn('["base_sha"]', summary_text)
    test.assertIn("stale head", summary_text)
    test.assertIn("stale base", summary_text)


def assert_author_contract(test: unittest.TestCase) -> None:
    agents = AGENTS.read_text()
    guide = AUTHOR_GUIDE.read_text()
    for text in (agents, guide):
        test.assertIn("PR Contract Acknowledgements", text)
        test.assertIn("PR Package Expansion", text)
        test.assertIn("exact head SHA", text)
        test.assertIn("candidate change", text)
        test.assertIn("inherited", text)
        test.assertIn("incomplete coverage", text)
        test.assertIn("base", text)
        test.assertIn("fresh", text)
    test.assertIn("gh workflow run pr-package-expansion.yml", guide)
    test.assertIn("-f pr_number=", guide)
    test.assertIn("-f expected_head_sha=", guide)


class PullRequestWorkflowRoutingTests(unittest.TestCase):
    def test_current_workflows_preserve_the_selected_contract(self) -> None:
        assert_ci_metadata_routing(self, yaml.safe_load(CI.read_text()))
        assert_acknowledgement_workflow(
            self, yaml.safe_load(ACKNOWLEDGEMENTS.read_text())
        )
        assert_manual_expansion_workflow(
            self, yaml.safe_load(EXPANSION.read_text())
        )
        assert_author_contract(self)

    def test_body_edits_cannot_reuse_required_context_names(self) -> None:
        workflow = yaml.safe_load(CI.read_text())
        for job_id in REQUIRED_IMPLEMENTATION_JOBS:
            mutated = copy.deepcopy(workflow)
            mutated["jobs"][job_id]["name"] = REQUIRED_IMPLEMENTATION_JOBS[job_id]
            with self.subTest(job=job_id), self.assertRaises(AssertionError):
                assert_ci_metadata_routing(self, mutated)

    def test_base_retarget_signal_and_dedicated_acknowledgements_are_required(self) -> None:
        workflow = yaml.safe_load(CI.read_text())
        mutated = copy.deepcopy(workflow)
        policy = next(
            step
            for step in mutated["jobs"]["changes"]["steps"]
            if step.get("id") == "event-policy"
        )
        policy["env"].pop("BASE_CHANGED")
        with self.assertRaises(AssertionError):
            assert_ci_metadata_routing(self, mutated)

        acknowledgements = yaml.safe_load(ACKNOWLEDGEMENTS.read_text())
        del acknowledgements["jobs"]["acknowledgements"]
        with self.assertRaises((AssertionError, KeyError)):
            assert_acknowledgement_workflow(self, acknowledgements)

    def test_automatic_expansion_or_an_unbound_dispatch_is_rejected(self) -> None:
        workflow = yaml.safe_load(CI.read_text())
        workflow["jobs"]["package-expansion-shard"] = {}
        with self.assertRaises(AssertionError):
            assert_ci_metadata_routing(self, workflow)

        manual = yaml.safe_load(EXPANSION.read_text())
        manual = copy.deepcopy(manual)
        del actions_events(manual)["workflow_dispatch"]["inputs"]["expected_head_sha"]
        with self.assertRaises(AssertionError):
            assert_manual_expansion_workflow(self, manual)


if __name__ == "__main__":
    unittest.main()
