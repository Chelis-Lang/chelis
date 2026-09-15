#!/usr/bin/env python3
"""Structural contract for the scheduled standing rejection-authority canary."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
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
CONCURRENCY_GROUP = "loud-unsupported-standing-liveness-main"
SUPPRESSING_SHELL = "bash -c 'source {0} || true'"


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


def _assert_no_run_shell_default(
    test: unittest.TestCase,
    owner: dict,
) -> None:
    """Reject an inherited shell without forbidding other run defaults."""
    defaults = owner.get("defaults")
    if defaults is None:
        return
    test.assertIsInstance(defaults, dict)
    run_defaults = defaults.get("run")
    if run_defaults is None:
        return
    test.assertIsInstance(run_defaults, dict)
    test.assertNotIn("shell", run_defaults)


class PullRequestLivenessTests(unittest.TestCase):
    """PR row admission and the standing canary must keep distinct modes."""

    def assert_pr_contract(self, workflow: dict) -> None:
        job = workflow["jobs"]["rejection-authority-liveness"]
        self.assertIn("github.event_name == 'pull_request'", job["if"])
        self.assertIn("rejection_authority_changed", job["if"])
        _assert_no_failure_suppression(self, job)
        _assert_no_run_shell_default(self, workflow)
        _assert_no_run_shell_default(self, job)
        checkouts = [s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout@")]
        self.assertEqual(len(checkouts), 1)
        self.assertEqual(checkouts[0].get("with", {}).get("fetch-depth"), 0)
        self.assertEqual(
            checkouts[0].get("with", {}).get("ref"),
            "${{ needs.changes.outputs.candidate_sha }}",
        )
        commands = [s for s in job["steps"] if "run" in s and "validate_rejection_issue_manifest.py" in s["run"]]
        self.assertEqual(len(commands), 1)
        step = commands[0]
        self.assertEqual(step["run"], COMMAND + ' --pr-head "$PR_HEAD"')
        self.assertEqual(step["env"]["PR_HEAD"], "${{ github.event.pull_request.head.sha }}")
        self.assertEqual(step["env"]["GH_TOKEN"], "${{ github.token }}")
        self.assertNotIn("if", step)
        self.assertNotIn("shell", step)

    def workflow(self) -> dict:
        return load_workflow((ROOT / ".github/workflows/ci.yml").read_text())

    def test_pr_job_uses_full_merge_checkout_and_exact_event_head(self) -> None:
        self.assert_pr_contract(self.workflow())

    def test_rejects_missing_history_wrong_checkout_or_wrong_head(self) -> None:
        for mutation in ["shallow", "head checkout", "wrong event"]:
            with self.subTest(mutation=mutation):
                workflow = self.workflow()
                steps = workflow["jobs"]["rejection-authority-liveness"]["steps"]
                if mutation == "shallow":
                    steps[0]["with"] = {"fetch-depth": 1}
                elif mutation == "head checkout":
                    steps[0]["with"] = {
                        "fetch-depth": 0,
                        "ref": "${{ github.event.pull_request.head.sha }}",
                    }
                else:
                    steps[-1]["env"]["PR_HEAD"] = "${{ github.sha }}"
                with self.assertRaises(AssertionError):
                    self.assert_pr_contract(workflow)

    def test_rejects_default_standing_mode_or_suppressed_failure_on_pr(self) -> None:
        for mutation in ["standing", "shell suppression", "continue", "skip"]:
            with self.subTest(mutation=mutation):
                workflow = self.workflow()
                step = workflow["jobs"]["rejection-authority-liveness"]["steps"][-1]
                if mutation == "standing":
                    step["run"] = COMMAND
                elif mutation == "shell suppression":
                    step["run"] += " || true"
                elif mutation == "continue":
                    step["continue-on-error"] = True
                else:
                    step["if"] = False
                with self.assertRaises(AssertionError):
                    self.assert_pr_contract(workflow)


def execute_report_reconciliation(script: str) -> dict:
    """Run the exact github-script body against a paginated in-memory issue API."""
    harness = f"""
const script = {json.dumps(script)};
const AsyncFunction = Object.getPrototypeOf(async function() {{}}).constructor;
const runScript = new AsyncFunction("github", "context", "core", "process", script);
const title = {json.dumps(TRACKING_TITLE)};
const state = [
  {{number: 11, title, state: "open"}},
  {{number: 12, title, state: "open"}},
  {{number: 13, title, state: "open"}},
  {{number: 14, title, state: "open", pull_request: {{}}}},
  {{number: 15, title: "Unrelated", state: "open"}},
];
let nextNumber = 100;
const comments = [];
const issues = {{
  listForRepo: async () => ({{
    data: state.filter(issue => issue.state === "open"),
  }}),
  create: async args => {{
    const issue = {{
      number: nextNumber++,
      title: args.title,
      state: "open",
      labels: args.labels,
    }};
    state.push(issue);
    return {{data: issue}};
  }},
  createComment: async args => {{
    comments.push({{number: args.issue_number, body: args.body}});
    return {{data: {{}}}};
  }},
  update: async args => {{
    const issue = state.find(candidate => candidate.number === args.issue_number);
    if (!issue) throw new Error(`missing issue ${{args.issue_number}}`);
    issue.state = args.state;
    return {{data: issue}};
  }},
}};
const github = {{
  paginate: async (method, args) => (await method(args)).data,
  rest: {{issues}},
}};
const context = {{
  repo: {{owner: "Chelis-Lang", repo: "chelis"}},
  serverUrl: "https://github.example",
  runId: 900,
}};
const core = {{info: () => {{}}}};
async function run(result, runId) {{
  context.runId = runId;
  await runScript(github, context, core, {{env: {{RESULT: result}}}});
}}
function openExactIssues() {{
  return state
    .filter(issue =>
      issue.state === "open" &&
      !issue.pull_request &&
      issue.title === title
    )
    .map(issue => issue.number)
    .sort((a, b) => a - b);
}}
(async () => {{
  await run("failure", 901);
  const afterFailure = openExactIssues();
  await run("success", 902);
  const afterSuccess = openExactIssues();
  await run("cancelled", 903);
  const afterReopen = openExactIssues();
  console.log(JSON.stringify({{
    afterFailure,
    afterSuccess,
    afterReopen,
    comments,
    state,
  }}));
}})().catch(error => {{
  console.error(error.stack || String(error));
  process.exit(1);
}});
"""
    result = subprocess.run(
        ["node", "-e", harness],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return json.loads(result.stdout)


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
    _assert_no_run_shell_default(test, workflow)
    test.assertEqual(
        workflow.get("concurrency"),
        {
            "group": CONCURRENCY_GROUP,
            "cancel-in-progress": False,
        },
    )
    test.assertEqual(set(workflow["jobs"]), {CANARY_JOB, REPORT_JOB})

    canary = workflow["jobs"][CANARY_JOB]
    _assert_no_run_shell_default(test, canary)
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
    test.assertNotIn("shell", command_step)

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
    test.assertNotIn("if", report_step)
    script = report_step["with"]["script"]
    for required in (
        f"const title = '{TRACKING_TITLE}';",
        "const passed = process.env.RESULT === 'success';",
        "const matches = (await github.paginate(",
        "github.rest.issues.listForRepo",
        ".filter(issue => !issue.pull_request && issue.title === title);",
        "const [canonical, ...duplicates] = matches;",
        "if (!passed) {",
        "github.rest.issues.create({",
        "labels: ['nightly-failure']",
        "for (const duplicate of duplicates) {",
        "} else {",
        "for (const issue of matches) {",
        "github.rest.issues.createComment({",
        "github.rest.issues.update({",
        "state: 'closed'",
    ):
        test.assertIn(required, script)
    test.assertNotIn(".find(", script)
    test.assertNotIn("search.issuesAndPullRequests", script)
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

    def test_workflow_concurrency_is_constant_and_never_cancels_running_work(self) -> None:
        mutations = []

        missing = copy.deepcopy(self.workflow)
        del missing["concurrency"]
        mutations.append(missing)

        per_run = copy.deepcopy(self.workflow)
        per_run["concurrency"]["group"] = "${{ github.run_id }}"
        mutations.append(per_run)

        per_ref = copy.deepcopy(self.workflow)
        per_ref["concurrency"]["group"] = (
            "loud-unsupported-${{ github.ref }}"
        )
        mutations.append(per_ref)

        cancelling = copy.deepcopy(self.workflow)
        cancelling["concurrency"]["cancel-in-progress"] = True
        mutations.append(cancelling)

        for index, mutated in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(AssertionError):
                self.assert_contract(mutated)

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
        for change in ("noop", "duplicate", "suppressing-shell"):
            with self.subTest(change=change):
                mutated = copy.deepcopy(self.workflow)
                steps = mutated["jobs"][CANARY_JOB]["steps"]
                command_step = next(step for step in steps if step.get("run") == COMMAND)
                if change == "noop":
                    command_step["run"] = f"true # {COMMAND}"
                elif change == "duplicate":
                    steps.append(copy.deepcopy(command_step))
                else:
                    command_step["shell"] = SUPPRESSING_SHELL
                with self.assertRaises(AssertionError):
                    self.assert_contract(mutated)

    def test_validator_cannot_inherit_suppressing_run_shell_defaults(self) -> None:
        for owner in ("workflow", CANARY_JOB):
            with self.subTest(owner=owner):
                mutated = copy.deepcopy(self.workflow)
                target = mutated if owner == "workflow" else mutated["jobs"][owner]
                target["defaults"] = {"run": {"shell": SUPPRESSING_SHELL}}
                with self.assertRaises(AssertionError):
                    self.assert_contract(mutated)

    def test_non_shell_run_defaults_remain_allowed(self) -> None:
        mutated = copy.deepcopy(self.workflow)
        mutated["defaults"] = {"run": {"working-directory": "."}}
        mutated["jobs"][CANARY_JOB]["defaults"] = {
            "run": {"working-directory": "."}
        }
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

        skipped_step = copy.deepcopy(self.workflow)
        skipped_step["jobs"][REPORT_JOB]["steps"][0]["if"] = (
            "env.RESULT == 'success'"
        )
        mutations.append(skipped_step)

        for index, mutated in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(AssertionError):
                self.assert_contract(mutated)

    def test_tracking_issue_all_match_reconciliation_is_required(self) -> None:
        for fragment in (
            "const matches = (await github.paginate(",
            "github.rest.issues.listForRepo",
            ".filter(issue => !issue.pull_request && issue.title === title);",
            "const [canonical, ...duplicates] = matches;",
            "github.rest.issues.create({",
            "labels: ['nightly-failure']",
            "for (const duplicate of duplicates) {",
            "} else {",
            "for (const issue of matches) {",
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

    def test_exact_report_script_collapses_duplicates_and_closes_all_on_recovery(
        self,
    ) -> None:
        script = self.workflow["jobs"][REPORT_JOB]["steps"][0]["with"]["script"]
        result = execute_report_reconciliation(script)
        self.assertEqual(result["afterFailure"], [11])
        self.assertEqual(result["afterSuccess"], [])
        self.assertEqual(result["afterReopen"], [100])
        issue_states = {
            issue["number"]: issue["state"]
            for issue in result["state"]
            if issue["title"] == TRACKING_TITLE and "pull_request" not in issue
        }
        self.assertEqual(
            issue_states,
            {11: "closed", 12: "closed", 13: "closed", 100: "open"},
        )

    def test_non_gate_classification_is_mandatory(self) -> None:
        missing = set(gate_contract.NON_GATE_WORKFLOWS)
        missing.discard(WORKFLOW_NAME)
        with self.assertRaises(AssertionError):
            self.assert_contract(non_gate=missing)


if __name__ == "__main__":
    unittest.main()
