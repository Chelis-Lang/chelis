"""Lock metadata-only CI routing and the manual expansion interface."""

from __future__ import annotations

import copy
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github/workflows/ci.yml"
ACKNOWLEDGEMENTS = ROOT / ".github/workflows/pr-contract-acknowledgements.yml"
EXPANSION = ROOT / ".github/workflows/pr-package-expansion.yml"
RETARGET = ROOT / ".github/workflows/pr-base-retarget.yml"
RECEIPT = ROOT / ".github/workflows/pr-candidate-receipt.yml"
HULL = ROOT / ".github/workflows/conformance.yml"
AGENTS = ROOT / "AGENTS.md"
AUTHOR_GUIDE = ROOT / "docs/guard_changes_for_pr_authors.md"
CI_VALIDATION = ROOT / "docs/ci_validation.md"
REQUIRED_IMPLEMENTATION_JOBS = {
    "lint-and-unit": "Lint and Unit Tests (Linux)",
    "integration": "Integration Tests (Linux)",
    "backend-sanitizers": "Backend Sanitizers",
    "smt-build": "SMT Feature Build (Linux)",
    "docs": "Docs",
    "no-ai-authorship": "No AI authorship markers",
}
PREFLIGHT_GATED_JOBS = {
    "diagnostic-kind-oracle",
    "lint-rust",
    "script-unit",
    "smt-build",
    "smt-build-glibc231",
    "ci-fast",
    "integration-plan",
    "test-telemetry",
    "backend-sanitizers",
}
BOOTSTRAP_CONTRACT_MARKERS = {
    "scripts/ci_contract_paths.py",
    "git -C candidate show",
    "BASE_SHA",
}
RECEIPT_TRIGGER_WORKFLOWS = {
    "CI",
    "Hull Conformance",
    "Changelog",
    "PR Contract Acknowledgements",
    "PR Base Retarget Validation",
}
PACKAGE_EXPANSION_INVALIDATION_CLASSES = (
    "content change",
    "hand-resolved conflict",
    "base-changing rebase",
    "base-branch retarget",
)
PACKAGE_EXPANSION_INVALIDATION_CLAUSE_SUFFIXES = {
    AGENTS: "creates a new synthetic candidate and requires a fresh dispatch.",
    AUTHOR_GUIDE: (
        "requires a fresh package-expansion dispatch because the workflow "
        "validates the exact synthetic candidate."
    ),
}


def actions_events(workflow: dict) -> dict:
    return workflow.get("on", workflow.get(True))


def remove_from_invalidation_clause(
    raw_text: str, document: Path, requirement: str
) -> str:
    text = " ".join(raw_text.split())
    suffix = PACKAGE_EXPANSION_INVALIDATION_CLAUSE_SUFFIXES[document]
    suffix_start = text.index(suffix)
    clause_start = text.rfind(". ", 0, suffix_start) + 2
    clause = text[clause_start : suffix_start + len(suffix)]
    mutated_clause = clause.replace(requirement, "", 1)
    if mutated_clause == clause:
        raise AssertionError(f"{requirement!r} is absent from the invalidation clause")
    return text[:clause_start] + mutated_clause + text[suffix_start + len(suffix) :]


def package_expansion_invalidation_clause(
    test: unittest.TestCase, document: Path, text: str
) -> str:
    suffix = PACKAGE_EXPANSION_INVALIDATION_CLAUSE_SUFFIXES[document]
    test.assertEqual(
        text.count(suffix),
        1,
        f"{document.name}: expected one package-expansion invalidation clause",
    )
    suffix_start = text.index(suffix)
    sentence_boundary = text.rfind(". ", 0, suffix_start)
    test.assertNotEqual(
        sentence_boundary,
        -1,
        f"{document.name}: package-expansion invalidation clause has no boundary",
    )
    clause = text[sentence_boundary + 2 : suffix_start + len(suffix)]
    test.assertTrue(
        clause.startswith("A "),
        f"{document.name}: malformed package-expansion invalidation clause",
    )
    return clause


def assert_ci_metadata_routing(test: unittest.TestCase, workflow: dict) -> None:
    test.assertEqual(workflow["permissions"], {"contents": "read"})
    events = actions_events(workflow)
    test.assertEqual(
        set(events["pull_request"]["types"]),
        {"opened", "synchronize", "reopened"},
    )
    inputs = events["workflow_dispatch"]["inputs"]
    test.assertEqual(
        set(inputs),
        {
            "pr_number",
            "expected_head_sha",
            "expected_base_sha",
            "retarget_token",
        },
    )
    concurrency = str(workflow["concurrency"])
    test.assertIn("inputs.pr_number", concurrency)
    test.assertIn("implementation", concurrency)

    changes = workflow["jobs"]["changes"]
    test.assertEqual(
        changes["permissions"],
        {
            "contents": "read",
            "issues": "read",
            "pull-requests": "read",
        },
    )
    test.assertIn("candidate_sha", changes["outputs"])
    test.assertIn("candidate_preflight", changes["outputs"])
    test.assertEqual(
        changes["outputs"]["ci_contract_changed"],
        "${{ steps.candidate-preflight.outputs.ci_contract_changed }}",
    )
    checkout = next(
        step
        for step in changes["steps"]
        if step.get("name") == "Checkout exact candidate"
    )
    test.assertIn("refs/pull/", str(checkout))
    test.assertIn("inputs.pr_number", str(checkout))
    validation = next(
        step for step in changes["steps"]
        if "ci_validate_pr_candidate.py" in step.get("run", "")
    )
    test.assertEqual(validation["if"], "github.event_name == 'workflow_dispatch'")
    test.assertIn("../trusted/scripts/ci_validate_pr_candidate.py", validation["run"])
    test.assertIn("--expected-base-sha", validation["run"])
    test.assertIn("--validate-checkout", validation["run"])
    trusted = next(
        step for step in changes["steps"]
        if step.get("name") == "Checkout trusted retarget validator"
    )
    test.assertEqual(trusted["with"]["ref"], "${{ inputs.expected_base_sha }}")
    test.assertLess(changes["steps"].index(trusted), changes["steps"].index(checkout))
    test.assertLess(
        changes["steps"].index(checkout), changes["steps"].index(validation)
    )
    lifecycle = next(
        step
        for step in changes["steps"]
        if "ci_candidate_lifecycle.py" in step.get("run", "")
    )
    test.assertTrue(lifecycle["continue-on-error"])
    test.assertIn("gh api", lifecycle["run"])
    test.assertIn("--current-pr", lifecycle["run"])
    test.assertIn("--target-tip", lifecycle["run"])
    test.assertIn("--timeline", lifecycle["run"])
    contract = next(
        step
        for step in changes["steps"]
        if "scripts.test_ci_candidate_lifecycle" in step.get("run", "")
    )
    test.assertTrue(contract["continue-on-error"])
    test.assertIn("scripts.test_check_agent_skills", contract["run"])
    test.assertIn("scripts.test_ci_candidate_identity", contract["run"])
    test.assertIn("scripts.test_ci_candidate_receipt", contract["run"])
    test.assertIn("scripts.test_ci_change_owned", contract["run"])
    test.assertIn("scripts.test_ci_contract_paths", contract["run"])
    test.assertIn("scripts.test_ci_validate_pr_candidate", contract["run"])
    test.assertIn("scripts.test_regenerate_conformance_assets", contract["run"])
    test.assertIn("scripts.test_gate.DocsOnlySkipTests", contract["run"])
    test.assertIn("scripts.test_gate.CiParityTests", contract["run"])
    test.assertIn(
        "scripts.test_gate.RejectionAuthorityPrBoundaryTests",
        contract["run"],
    )
    test.assertIn(
        "steps.ci-contract-bootstrap.outputs.ci_contract_changed == 'true'",
        contract["if"],
    )
    bootstrap = next(
        step
        for step in changes["steps"]
        if step.get("id") == "ci-contract-bootstrap"
    )
    test.assertEqual(bootstrap["if"], "always()")
    for marker in BOOTSTRAP_CONTRACT_MARKERS:
        test.assertIn(marker, bootstrap["run"])
    record = next(
        step
        for step in changes["steps"]
        if step.get("id") == "candidate-preflight"
    )
    test.assertIn("candidate_preflight=", record["run"])
    test.assertIn("ci_contract_changed=", record["run"])
    test.assertIn('if [ "$lifecycle" != "success" ]', record["run"])
    test.assertIn('if [ "$detected_contract" = "true" ]', record["run"])
    test.assertIn('[ "$bootstrap_contract" = "true" ]', record["run"])
    identity = next(
        step
        for step in changes["steps"]
        if step.get("name") == "Record immutable candidate identity"
    )
    test.assertEqual(
        identity["if"],
        "github.event_name == 'pull_request' && steps.candidate-preflight.outputs.candidate_preflight == 'success'",
    )
    test.assertEqual(identity["working-directory"], "candidate")
    test.assertIn("scripts/ci_candidate_identity.py", identity["run"])
    test.assertIn("--workflow-file ci.yml", identity["run"])
    test.assertIn("github.event.pull_request.base.sha", identity["run"])
    upload = next(
        step
        for step in changes["steps"]
        if step.get("name") == "Upload immutable candidate identity"
    )
    test.assertEqual(upload["if"], identity["if"])
    test.assertEqual(upload["with"]["name"], "candidate-identity-ci")
    test.assertEqual(upload["with"]["if-no-files-found"], "error")
    test.assertIn("candidate-identity.json", upload["with"]["path"])
    test.assertIn('if [ "$contract_changed" = "true" ]', record["run"])
    docs = workflow["jobs"]["docs"]
    test.assertIn("candidate_preflight", str(docs["steps"]))
    for job_id in PREFLIGHT_GATED_JOBS:
        test.assertIn(
            "needs.changes.outputs.candidate_preflight == 'success'",
            workflow["jobs"][job_id]["if"],
        )

    for job_id, required_name in REQUIRED_IMPLEMENTATION_JOBS.items():
        job = workflow["jobs"][job_id]
        test.assertIn("changes", job["needs"])
        test.assertEqual(job["name"], required_name)
        if job_id in {"backend-sanitizers", "smt-build"}:
            test.assertIn(
                "needs.changes.outputs.candidate_preflight == 'success'",
                job["if"],
            )

    test.assertNotIn("package-expansion-shard", workflow["jobs"])
    test.assertNotIn("package-expansion-summary", workflow["jobs"])


def assert_retarget_workflow(test: unittest.TestCase, workflow: dict) -> None:
    test.assertNotIn("concurrency", workflow)
    events = actions_events(workflow)
    test.assertEqual(
        events,
        {
            "pull_request": {
                "types": ["opened", "synchronize", "reopened"],
            },
            "pull_request_target": {"types": ["edited"]},
        },
    )
    jobs = workflow["jobs"]
    ordinary = jobs["ordinary-candidate"]
    test.assertEqual(ordinary["name"], "PR Base Retarget Validation")
    test.assertEqual(
        ordinary["if"], "github.event_name == 'pull_request'"
    )
    coordinator = jobs["base-retarget"]
    test.assertIn("github.event.changes.base", coordinator["if"])
    test.assertEqual(
        coordinator["concurrency"],
        {
            "group": "pr-base-retarget-${{ github.event.pull_request.number }}",
            "cancel-in-progress": True,
        },
    )
    test.assertEqual(
        coordinator["permissions"],
        {
            "actions": "write",
            "checks": "write",
            "contents": "read",
            "pull-requests": "read",
        },
    )
    text = str(coordinator)
    test.assertIn("scripts/ci_retarget_validation.py", text)
    test.assertIn("github.event.pull_request.head.sha", text)
    test.assertIn("github.event.pull_request.base.sha", text)
    test.assertIn("github.event.pull_request.base.ref", text)
    test.assertNotIn("github.event.pull_request.body", text)


def assert_hull_retarget_dispatch(test: unittest.TestCase, workflow: dict) -> None:
    events = actions_events(workflow)
    test.assertEqual(
        set(events["pull_request"]["types"]),
        {"opened", "synchronize", "reopened"},
    )
    test.assertIn("workflow_dispatch", events)
    test.assertEqual(
        set(events["workflow_dispatch"]["inputs"]),
        {
            "pr_number",
            "expected_head_sha",
            "expected_base_sha",
            "retarget_token",
        },
    )
    changes = workflow["jobs"]["changes"]
    test.assertIn("candidate_sha", changes["outputs"])
    test.assertIn("candidate_preflight", changes["outputs"])
    test.assertEqual(
        changes["outputs"]["ci_contract_changed"],
        "${{ steps.candidate-preflight.outputs.ci_contract_changed }}",
    )
    test.assertIn("ci_validate_pr_candidate.py", str(changes))
    test.assertIn("ci_candidate_lifecycle.py", str(changes))
    test.assertIn("scripts.test_ci_candidate_lifecycle", str(changes))
    bootstrap = next(
        step
        for step in changes["steps"]
        if step.get("id") == "ci-contract-bootstrap"
    )
    test.assertEqual(bootstrap["if"], "always()")
    for marker in BOOTSTRAP_CONTRACT_MARKERS:
        test.assertIn(marker, bootstrap["run"])
    contract = next(
        step for step in changes["steps"] if step.get("id") == "ci-contract"
    )
    test.assertIn(
        "steps.ci-contract-bootstrap.outputs.ci_contract_changed == 'true'",
        contract["if"],
    )
    identity = next(
        step
        for step in changes["steps"]
        if step.get("name") == "Record immutable candidate identity"
    )
    test.assertEqual(
        identity["if"],
        "github.event_name == 'pull_request' && steps.candidate-preflight.outputs.candidate_preflight == 'success'",
    )
    test.assertEqual(identity["working-directory"], "candidate")
    test.assertIn("scripts/ci_candidate_identity.py", identity["run"])
    test.assertIn("--workflow-file conformance.yml", identity["run"])
    upload = next(
        step
        for step in changes["steps"]
        if step.get("name") == "Upload immutable candidate identity"
    )
    test.assertEqual(upload["if"], identity["if"])
    test.assertEqual(upload["with"]["name"], "candidate-identity-hull")
    test.assertEqual(upload["with"]["if-no-files-found"], "error")
    conformance = workflow["jobs"]["conformance"]
    test.assertIn(
        "needs.changes.outputs.candidate_preflight != 'success'",
        conformance["if"],
    )
    preflight = conformance["steps"][0]
    test.assertEqual(
        preflight["name"],
        "Require candidate lifecycle and CI contract preflight",
    )
    test.assertIn("CANDIDATE_PREFLIGHT", preflight["env"])
    test.assertIn('!= "success"', preflight["run"])
    checkout = next(
        step for step in conformance["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )
    test.assertEqual(
        checkout["with"]["ref"],
        "${{ needs.changes.outputs.candidate_sha }}",
    )


def assert_acknowledgement_workflow(test: unittest.TestCase, workflow: dict) -> None:
    events = actions_events(workflow)
    test.assertEqual(
        set(events["pull_request"]["types"]),
        {"opened", "synchronize", "reopened", "edited"},
    )
    test.assertIn("pull_request.number", str(workflow["concurrency"]))
    job = workflow["jobs"]["acknowledgements"]
    test.assertEqual(job["name"], "PR Contract Acknowledgements")
    test.assertEqual(
        job["permissions"],
        {
            "contents": "read",
            "issues": "read",
            "pull-requests": "read",
        },
    )
    text = str(job)
    test.assertNotIn("cargo ", text)
    test.assertNotIn("rust-toolchain", text)
    test.assertIn("fetch-depth", text)
    test.assertIn("phase3_test_change_report.py", text)
    test.assertIn("phase4b_change_report.py", text)
    test.assertEqual(text.count("--require-acknowledgement"), 2)
    test.assertIn("github.event.pull_request.head.sha", text)
    test.assertIn("github.event.pull_request.body", text)
    test.assertIn("ci_candidate_lifecycle.py", text)
    test.assertIn("--current-pr", text)
    test.assertIn("--target-tip", text)
    test.assertIn("--timeline", text)
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
    test.assertEqual(
        set(planner["outputs"]),
        {"base_ref", "candidate_sha"},
    )
    test.assertIn("pull-requests", str(planner["permissions"]))
    test.assertIn("refs/pull/", planner_text)
    test.assertIn("expected_head_sha", planner_text)
    test.assertIn("scripts/ci_validate_pr_candidate.py", planner_text)
    test.assertNotIn("--expected-base-sha", planner_text)
    test.assertIn("--event-name", planner_text)
    test.assertIn("pull_request", planner_text)
    test.assertIn("--pr-head", planner_text)
    planner_steps = planner["steps"]
    checkouts = [
        index
        for index, step in enumerate(planner_steps)
        if step.get("uses", "").startswith("actions/checkout@")
    ]
    validator = next(
        index
        for index, step in enumerate(planner_steps)
        if "scripts/ci_validate_pr_candidate.py" in step.get("run", "")
    )
    test.assertEqual(len(checkouts), 2)
    test.assertLess(checkouts[0], validator)
    test.assertLess(validator, checkouts[1])
    test.assertEqual(
        planner_steps[checkouts[0]].get("with", {}).get("ref"),
        "${{ github.sha }}",
    )
    test.assertEqual(
        planner_steps[checkouts[0]].get("with", {}).get("path"),
        "trusted",
    )
    test.assertEqual(
        planner_steps[checkouts[1]].get("with", {}).get("ref"),
        "refs/pull/${{ inputs.pr_number }}/merge",
    )
    test.assertEqual(
        planner_steps[checkouts[1]].get("with", {}).get("path"),
        "candidate",
    )
    candidate = next(
        step for step in planner_steps if step.get("id") == "candidate"
    )
    test.assertIn("candidate_sha=", candidate["run"])
    test.assertEqual(candidate["working-directory"], "candidate")
    identity = next(
        step for step in planner_steps if step.get("id") == "pr-identity"
    )
    test.assertIn("trusted/scripts/ci_validate_pr_candidate.py", identity["run"])
    test.assertIn("--github-output \"$GITHUB_OUTPUT\"", identity["run"])

    worker = jobs["package-expansion-shard"]
    test.assertEqual(worker["needs"], ["integration-plan"])
    test.assertEqual(worker["strategy"]["matrix"]["shard"], [0, 1, 2, 3])
    test.assertIn("--lane package-expansion", str(worker))
    worker_checkout = next(
        step
        for step in worker["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )
    test.assertEqual(
        worker_checkout["with"]["ref"],
        "${{ needs.integration-plan.outputs.candidate_sha }}",
    )
    summary = jobs["package-expansion-summary"]
    test.assertEqual(
        summary["needs"], ["integration-plan", "package-expansion-shard"]
    )
    test.assertIn("always()", summary["if"])
    summary_text = str(summary)
    test.assertIn("--lane package-expansion", summary_text)
    test.assertNotIn("--required", str(summary))
    test.assertIn("scripts/ci_validate_pr_candidate.py", summary_text)
    test.assertIn("--plan target/integration-change/plan.json", summary_text)
    test.assertIn("--expected-base-ref", summary_text)
    test.assertIn("--checkout-base-sha", summary_text)
    test.assertNotIn("--expected-base-sha", summary_text)
    test.assertIn("No validated plan artifact was available.", summary_text)
    test.assertGreaterEqual(summary_text.count("exit 1"), 2)
    workflow_text = str(workflow)
    test.assertEqual(
        workflow_text.count("scripts/ci_validate_pr_candidate.py"),
        2,
    )
    summary_checkout = next(
        step
        for step in summary["steps"]
        if step.get("with", {}).get("path") == "candidate"
    )
    test.assertEqual(
        summary_checkout["with"]["ref"],
        "${{ needs.integration-plan.outputs.candidate_sha }}",
    )
    trusted_summary_checkout = next(
        step
        for step in summary["steps"]
        if step.get("with", {}).get("path") == "trusted"
    )
    test.assertEqual(
        trusted_summary_checkout["with"]["ref"],
        "${{ github.sha }}",
    )


def assert_candidate_receipt_workflow(
    test: unittest.TestCase, workflow: dict
) -> None:
    events = actions_events(workflow)
    test.assertEqual(set(events), {"workflow_run"})
    trigger = events["workflow_run"]
    test.assertEqual(set(trigger["workflows"]), RECEIPT_TRIGGER_WORKFLOWS)
    test.assertEqual(trigger["types"], ["completed"])
    test.assertEqual(workflow["permissions"], {"contents": "read"})
    test.assertIn(
        "github.event.workflow_run.head_sha",
        str(workflow["concurrency"]),
    )
    job = workflow["jobs"]["collect"]
    test.assertEqual(
        job["permissions"],
        {
            "actions": "read",
            "checks": "read",
            "contents": "read",
            "pull-requests": "read",
        },
    )
    test.assertIn(
        "github.event.workflow_run.event == 'pull_request'",
        job["if"],
    )
    test.assertIn(
        "github.event.workflow_run.head_repository.full_name == github.repository",
        job["if"],
    )
    checkouts = [
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    ]
    test.assertEqual(len(checkouts), 1)
    checkout = checkouts[0]
    test.assertEqual(
        checkout["with"]["ref"],
        "${{ github.event.repository.default_branch }}",
    )
    test.assertEqual(checkout["with"]["path"], "trusted")
    test.assertEqual(checkout["with"]["fetch-depth"], 0)
    text = str(job)
    test.assertNotIn("refs/pull/", text)
    test.assertNotIn("actions/checkout@v6", text)
    collect = next(
        step
        for step in job["steps"]
        if "ci_candidate_receipt.py" in step.get("run", "")
    )
    test.assertIn("trusted/scripts/ci_candidate_receipt.py", collect["run"])
    test.assertIn("github.event.workflow_run.id", str(collect))
    test.assertIn("github.run_id", str(collect))
    test.assertIn("GH_TOKEN", collect["env"])
    upload = next(
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    test.assertEqual(
        upload["with"]["name"],
        "pr-candidate-receipt-${{ github.event.workflow_run.head_sha }}",
    )
    test.assertEqual(upload["with"]["if-no-files-found"], "ignore")
    test.assertEqual(upload["with"]["retention-days"], 30)


def assert_candidate_receipt_docs(test: unittest.TestCase) -> None:
    text = " ".join(CI_VALIDATION.read_text().split())
    for requirement in (
        "default branch's `workflow_run` collector",
        "merge-base(base, head)..head",
        "same synthetic candidate",
        "expected workflow file and exact job id",
        "shadow evidence",
        "does not skip, cancel, or satisfy any required check",
    ):
        test.assertIn(requirement, text)


def assert_author_contract(test: unittest.TestCase) -> None:
    documents = {
        AGENTS: AGENTS.read_text(),
        AUTHOR_GUIDE: AUTHOR_GUIDE.read_text(),
    }
    for document, raw_text in documents.items():
        text = " ".join(raw_text.split())
        test.assertIn("PR Contract Acknowledgements", text)
        test.assertIn("PR Base Retarget Validation", text)
        test.assertIn("PR Package Expansion", text)
        test.assertIn("exact head SHA", text)
        invalidation_clause = package_expansion_invalidation_clause(
            test, document, text
        )
        for requirement in PACKAGE_EXPANSION_INVALIDATION_CLASSES:
            test.assertIn(requirement, invalidation_clause)
        test.assertIn("inherited", text)
        test.assertIn("incomplete coverage", text)
        test.assertIn("base", text)
        test.assertIn("fresh", text)
        test.assertIn("initial review candidate", text)
        test.assertIn("one consolidated repair candidate", text)
        test.assertIn("Candidate-base-update:", text)
        test.assertIn("Candidate-history-rewrite:", text)
        test.assertIn("before the first push", text)
        test.assertIn("A rebase does not by itself require a fresh round.", text)
        test.assertIn("hand-resolved intersection", text)
        test.assertIn(
            "Later movement of the same target branch does not invalidate",
            text,
        )
        test.assertIn("required Docs and Hull contexts", text)
    guide = documents[AUTHOR_GUIDE]
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
        assert_retarget_workflow(self, yaml.safe_load(RETARGET.read_text()))
        assert_hull_retarget_dispatch(self, yaml.safe_load(HULL.read_text()))
        assert_candidate_receipt_workflow(
            self, yaml.safe_load(RECEIPT.read_text())
        )
        assert_candidate_receipt_docs(self)
        assert_author_contract(self)

    def test_body_edits_cannot_reenter_compiler_ci(self) -> None:
        workflow = yaml.safe_load(CI.read_text())
        actions_events(workflow)["pull_request"]["types"].append("edited")
        with self.assertRaises(AssertionError):
            assert_ci_metadata_routing(self, workflow)

    def test_expensive_jobs_fail_closed_when_preflight_output_is_missing(
        self,
    ) -> None:
        for job_id in PREFLIGHT_GATED_JOBS:
            workflow = copy.deepcopy(yaml.safe_load(CI.read_text()))
            workflow["jobs"][job_id]["if"] = workflow["jobs"][job_id][
                "if"
            ].replace(
                "needs.changes.outputs.candidate_preflight == 'success' && ",
                "",
                1,
            )
            with self.subTest(job=job_id), self.assertRaises(AssertionError):
                assert_ci_metadata_routing(self, workflow)

    def test_candidate_detector_cannot_disable_its_own_contract_suite(self) -> None:
        for workflow_path, assertion in (
            (CI, assert_ci_metadata_routing),
            (HULL, assert_hull_retarget_dispatch),
        ):
            workflow = copy.deepcopy(yaml.safe_load(workflow_path.read_text()))
            bootstrap = next(
                step
                for step in workflow["jobs"]["changes"]["steps"]
                if step.get("id") == "ci-contract-bootstrap"
            )
            bootstrap["run"] = bootstrap["run"].replace(
                "scripts/ci_contract_paths.py",
                "scripts/no-contract-paths.py",
            )
            with self.subTest(workflow=workflow_path.name), self.assertRaises(
                AssertionError
            ):
                assertion(self, workflow)

    def test_ci_and_hull_share_the_same_bootstrap_path_boundary(self) -> None:
        workflows = [
            yaml.safe_load(path.read_text()) for path in (CI, HULL)
        ]
        bootstrap_runs = []
        for workflow in workflows:
            bootstrap_runs.append(
                next(
                    step
                    for step in workflow["jobs"]["changes"]["steps"]
                    if step.get("id") == "ci-contract-bootstrap"
                )["run"]
            )
        self.assertEqual(bootstrap_runs[0], bootstrap_runs[1])

    def test_author_contract_requires_each_explicit_invalidation_class(self) -> None:
        originals = {
            AGENTS: AGENTS.read_text(),
            AUTHOR_GUIDE: AUTHOR_GUIDE.read_text(),
        }
        for document in originals:
            for requirement in PACKAGE_EXPANSION_INVALIDATION_CLASSES:
                mutated = dict(originals)
                mutated[document] = (
                    f"Unrelated note retaining the words {requirement}.\n\n"
                    + remove_from_invalidation_clause(
                        mutated[document], document, requirement
                    )
                )
                with (
                    self.subTest(document=document.name, requirement=requirement),
                    patch.object(
                        Path,
                        "read_text",
                        autospec=True,
                        side_effect=lambda path: mutated[path],
                    ),
                    self.assertRaisesRegex(
                        AssertionError,
                        re.escape(f"'{requirement}' not found"),
                    ),
                ):
                    assert_author_contract(self)

    def test_base_retarget_signal_and_dedicated_acknowledgements_are_required(self) -> None:
        retarget = yaml.safe_load(RETARGET.read_text())
        mutated = copy.deepcopy(retarget)
        mutated["jobs"]["base-retarget"]["if"] = "false"
        with self.assertRaises(AssertionError):
            assert_retarget_workflow(self, mutated)

        acknowledgements = yaml.safe_load(ACKNOWLEDGEMENTS.read_text())
        del acknowledgements["jobs"]["acknowledgements"]
        with self.assertRaises((AssertionError, KeyError)):
            assert_acknowledgement_workflow(self, acknowledgements)

    def test_metadata_edit_cannot_cancel_a_live_base_retarget(self) -> None:
        retarget = yaml.safe_load(RETARGET.read_text())
        mutated = copy.deepcopy(retarget)
        mutated["concurrency"] = mutated["jobs"]["base-retarget"].pop(
            "concurrency"
        )
        with self.assertRaises(AssertionError):
            assert_retarget_workflow(self, mutated)

    def test_retargeted_candidate_cannot_receive_write_contents(self) -> None:
        workflow = yaml.safe_load(CI.read_text())
        workflow["permissions"]["contents"] = "write"
        with self.assertRaises(AssertionError):
            assert_ci_metadata_routing(self, workflow)

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

    def test_dispatch_cannot_drop_initial_or_final_candidate_validation(self) -> None:
        for job_id in ("integration-plan", "package-expansion-summary"):
            manual = copy.deepcopy(yaml.safe_load(EXPANSION.read_text()))
            for step in manual["jobs"][job_id]["steps"]:
                if "ci_validate_pr_candidate.py" in step.get("run", ""):
                    step["run"] = "true"
                    break
            with self.subTest(job=job_id), self.assertRaises(AssertionError):
                assert_manual_expansion_workflow(self, manual)

    def test_dispatch_cannot_validate_before_trusted_scripts_exist(self) -> None:
        manual = copy.deepcopy(yaml.safe_load(EXPANSION.read_text()))
        planner = manual["jobs"]["integration-plan"]
        planner["steps"][0], planner["steps"][1] = (
            planner["steps"][1],
            planner["steps"][0],
        )
        with self.assertRaises(AssertionError):
            assert_manual_expansion_workflow(self, manual)

    def test_receipt_collector_cannot_checkout_or_execute_pr_content(self) -> None:
        workflow = copy.deepcopy(yaml.safe_load(RECEIPT.read_text()))
        workflow["jobs"]["collect"]["steps"].insert(
            1,
            {
                "uses": "actions/checkout@v6",
                "with": {"ref": "refs/pull/1/merge", "path": "candidate"},
            },
        )
        with self.assertRaises(AssertionError):
            assert_candidate_receipt_workflow(self, workflow)

    def test_receipt_collector_cannot_gain_write_permissions(self) -> None:
        workflow = copy.deepcopy(yaml.safe_load(RECEIPT.read_text()))
        workflow["jobs"]["collect"]["permissions"]["actions"] = "write"
        with self.assertRaises(AssertionError):
            assert_candidate_receipt_workflow(self, workflow)


if __name__ == "__main__":
    unittest.main()
