#!/usr/bin/env python3
"""Structural contract for the scheduled standing rejection-authority canary."""

from __future__ import annotations

import copy
from pathlib import Path
import unittest

import yaml

from scripts import test_gate as gate_contract


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_NAME = "loud-unsupported-nightly.yml"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / WORKFLOW_NAME
CANARY_JOB = "standing-rejection-liveness"
REPORT_JOB = "report"
COMMAND = ".venv/bin/python scripts/validate_rejection_issue_manifest.py"
TRACKING_TITLE = "Nightly failing: Standing Rejection Authority Liveness"


class UniqueLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate mapping keys."""


def _construct_unique_mapping(
    loader: UniqueLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate workflow key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_workflow(source: str | None = None) -> dict:
    """Load the workflow as data without accepting duplicate-key drift."""
    text = WORKFLOW_PATH.read_text() if source is None else source
    workflow = yaml.load(text, Loader=UniqueLoader)
    if not isinstance(workflow, dict):
        raise AssertionError("workflow must be a mapping")
    return workflow


def _assert_no_failure_suppression(test: unittest.TestCase, job: dict) -> None:
    test.assertNotIn("continue-on-error", job)
    for step in job["steps"]:
        test.assertNotIn("continue-on-error", step)


def _assert_exact_command_once(test: unittest.TestCase, job: dict) -> None:
    runs = [step.get("run") for step in job["steps"] if "run" in step]
    test.assertEqual(runs.count(COMMAND), 1)


def assert_standing_canary_contract(
    test: unittest.TestCase,
    workflow: dict,
    non_gate_workflows: set[str],
) -> None:
    """Require the honest, fail-closed standing-canary slice."""
    triggers = workflow.get(True)  # PyYAML 1.1 decodes the Actions `on` key as True.
    test.assertEqual(
        triggers,
        {
            "schedule": [{"cron": "17 6 * * *"}],
            "workflow_dispatch": None,
        },
    )
    test.assertEqual(workflow.get("permissions"), {"contents": "read"})
    test.assertEqual(set(workflow["jobs"]), {CANARY_JOB, REPORT_JOB})

    canary = workflow["jobs"][CANARY_JOB]
    test.assertEqual(
        canary.get("name"),
        "Standing Rejection Authority Liveness (main)",
    )
    test.assertEqual(canary.get("runs-on"), "ubuntu-latest")
    test.assertEqual(canary.get("timeout-minutes"), 30)
    test.assertNotIn("if", canary)
    test.assertEqual(
        canary.get("permissions"),
        {"contents": "read", "issues": "read"},
    )
    _assert_no_failure_suppression(test, canary)
    test.assertEqual(
        canary["steps"][0],
        {
            "name": "Checkout main",
            "uses": "actions/checkout@v6",
            "with": {"ref": "main", "persist-credentials": False},
        },
    )
    _assert_exact_command_once(test, canary)
    command_step = next(step for step in canary["steps"] if step.get("run") == COMMAND)
    test.assertEqual(
        command_step.get("env"),
        {"GH_TOKEN": "${{ github.token }}"},
    )
    test.assertNotIn("if", command_step)

    report = workflow["jobs"][REPORT_JOB]
    test.assertEqual(report.get("name"), "Surface standing liveness status")
    test.assertEqual(report.get("needs"), CANARY_JOB)
    test.assertEqual(report.get("if"), "always()")
    test.assertEqual(report.get("runs-on"), "ubuntu-latest")
    test.assertEqual(report.get("permissions"), {"issues": "write"})
    _assert_no_failure_suppression(test, report)
    test.assertEqual(len(report["steps"]), 1)
    report_step = report["steps"][0]
    test.assertEqual(
        report_step.get("uses"),
        "actions/github-script@v7",
    )
    test.assertEqual(
        report_step.get("env"),
        {"RESULT": "${{ needs.standing-rejection-liveness.result }}"},
    )
    script = report_step["with"]["script"]
    for required in (
        f"const title = '{TRACKING_TITLE}';",
        "const passed = process.env.RESULT === 'success';",
        "if (!passed) {",
        "github.rest.search.issuesAndPullRequests",
        "github.rest.issues.create({",
        "labels: ['nightly-failure']",
        "} else if (existing) {",
        "github.rest.issues.createComment({",
        "github.rest.issues.update({",
        "state: 'closed'",
    ):
        test.assertIn(required, script)
    test.assertNotIn("process.env.RESULT === 'failure'", script)
    test.assertIn(WORKFLOW_NAME, non_gate_workflows)


class LoudUnsupportedNightlyWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = load_workflow()

    def assert_contract(self, workflow: dict | None = None, non_gate=None) -> None:
        assert_standing_canary_contract(
            self,
            self.workflow if workflow is None else workflow,
            (
                gate_contract.NON_GATE_WORKFLOWS
                if non_gate is None
                else non_gate
            ),
        )

    def test_current_workflow_delivers_the_standing_canary_only(self) -> None:
        self.assert_contract()

    def test_schedule_and_manual_only_trigger_policy_rejects_drift(self) -> None:
        for event in (
            "push",
            "pull_request",
            "pull_request_target",
            "workflow_run",
            "repository_dispatch",
        ):
            with self.subTest(event=event):
                mutated = copy.deepcopy(self.workflow)
                mutated[True][event] = None
                with self.assertRaises(AssertionError):
                    self.assert_contract(mutated)
        for missing in ("schedule", "workflow_dispatch"):
            with self.subTest(missing=missing):
                mutated = copy.deepcopy(self.workflow)
                del mutated[True][missing]
                with self.assertRaises(AssertionError):
                    self.assert_contract(mutated)
        mutated = copy.deepcopy(self.workflow)
        mutated[True]["schedule"] = [{"cron": "17 6 * * 0"}]
        with self.subTest(schedule="weekly"), self.assertRaises(AssertionError):
            self.assert_contract(mutated)

    def test_command_cannot_be_replaced_by_noop_or_duplicated(self) -> None:
        for change in ("noop", "duplicate"):
            with self.subTest(change=change):
                mutated = copy.deepcopy(self.workflow)
                steps = mutated["jobs"][CANARY_JOB]["steps"]
                command_step = next(step for step in steps if step.get("run") == COMMAND)
                if change == "noop":
                    command_step["run"] = f"true # {COMMAND}"
                else:
                    steps.append(copy.deepcopy(command_step))
                with self.assertRaises(AssertionError):
                    self.assert_contract(mutated)

    def test_issue_permissions_are_required_at_both_boundaries(self) -> None:
        for job, permission in (
            (CANARY_JOB, "issues"),
            (REPORT_JOB, "issues"),
        ):
            with self.subTest(job=job):
                mutated = copy.deepcopy(self.workflow)
                del mutated["jobs"][job]["permissions"][permission]
                with self.assertRaises(AssertionError):
                    self.assert_contract(mutated)

    def test_report_dependency_and_non_success_handling_cannot_drift(self) -> None:
        mutations = []
        wrong_need = copy.deepcopy(self.workflow)
        wrong_need["jobs"][REPORT_JOB]["needs"] = "unrelated-job"
        mutations.append(wrong_need)

        not_always = copy.deepcopy(self.workflow)
        not_always["jobs"][REPORT_JOB]["if"] = "success()"
        mutations.append(not_always)

        wrong_result = copy.deepcopy(self.workflow)
        wrong_result["jobs"][REPORT_JOB]["steps"][0]["env"]["RESULT"] = (
            "${{ needs.unrelated-job.result }}"
        )
        mutations.append(wrong_result)

        failure_only = copy.deepcopy(self.workflow)
        script = failure_only["jobs"][REPORT_JOB]["steps"][0]["with"]["script"]
        failure_only["jobs"][REPORT_JOB]["steps"][0]["with"]["script"] = script.replace(
            "if (!passed) {",
            "if (process.env.RESULT === 'failure') {",
            1,
        )
        mutations.append(failure_only)

        for index, mutated in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(AssertionError):
                self.assert_contract(mutated)

    def test_tracking_issue_open_and_close_paths_are_both_required(self) -> None:
        for fragment in (
            "github.rest.issues.create({",
            "labels: ['nightly-failure']",
            "} else if (existing) {",
            "github.rest.issues.createComment({",
            "github.rest.issues.update({",
            "state: 'closed'",
        ):
            with self.subTest(fragment=fragment):
                mutated = copy.deepcopy(self.workflow)
                script = mutated["jobs"][REPORT_JOB]["steps"][0]["with"]["script"]
                mutated["jobs"][REPORT_JOB]["steps"][0]["with"]["script"] = (
                    script.replace(fragment, "removed")
                )
                with self.assertRaises(AssertionError):
                    self.assert_contract(mutated)

    def test_non_gate_classification_is_mandatory(self) -> None:
        missing = set(gate_contract.NON_GATE_WORKFLOWS)
        missing.discard(WORKFLOW_NAME)
        with self.assertRaises(AssertionError):
            self.assert_contract(non_gate=missing)


if __name__ == "__main__":
    unittest.main()
