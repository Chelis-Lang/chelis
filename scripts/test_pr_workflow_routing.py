"""Lock metadata-only CI routing and the manual expansion interface."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

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
def actions_events(workflow: dict) -> dict:
    return workflow.get("on", workflow.get(True))


def run_rebase_selector(
    workflow: dict,
    *,
    event_name: str,
    action: str,
    lifecycle: str = "success",
) -> dict[str, str]:
    step = next(
        step
        for step in workflow["jobs"]["changes"]["steps"]
        if step.get("id") == "rebase-reuse"
    )
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "github-output"
        environment = os.environ.copy()
        environment.update(
            {
                "EVENT_NAME": event_name,
                "ACTION": action,
                "LIFECYCLE": lifecycle,
                "BEFORE": "",
                "GITHUB_OUTPUT": str(output),
            }
        )
        subprocess.run(
            ["bash", "-eu", "-o", "pipefail", "-c", step["run"]],
            cwd=ROOT,
            env=environment,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return dict(
            line.split("=", 1)
            for line in output.read_text().splitlines()
            if "=" in line
        )


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
            "actions": "read",
            "contents": "read",
            "issues": "read",
            "pull-requests": "read",
        },
    )
    test.assertIn("candidate_sha", changes["outputs"])
    test.assertIn("candidate_preflight", changes["outputs"])
    test.assertIn("rebase_lane", changes["outputs"])
    test.assertIn("rebase_before", changes["outputs"])
    test.assertIn("rebase_contract_changed", changes["outputs"])
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
    reuse_checkout = next(
        step
        for step in changes["steps"]
        if step.get("name") == "Checkout trusted rebase reuse verifier"
    )
    test.assertEqual(
        reuse_checkout["with"]["ref"],
        "${{ github.event.pull_request.base.sha }}",
    )
    test.assertEqual(reuse_checkout["with"]["path"], "trusted-rebase")
    reuse = next(
        step for step in changes["steps"] if step.get("id") == "rebase-reuse"
    )
    test.assertEqual(reuse["if"], "always()")
    test.assertIn(
        "../trusted-rebase/scripts/ci_rebase_reuse.py",
        reuse["run"],
    )
    test.assertIn("--current-pr", reuse["run"])
    test.assertIn("--timeline", reuse["run"])
    test.assertIn("--github-output", reuse["run"])
    test.assertIn("rebase_contract_changed=false", reuse["run"])
    test.assertIn("rebase_lane=ordinary", reuse["run"])
    publish = next(
        step
        for step in changes["steps"]
        if step.get("name") == "Publish targeted rebase decision for review"
    )
    test.assertEqual(publish["if"], "always()")
    test.assertIn("decision.json", publish["run"])
    test.assertIn("GITHUB_STEP_SUMMARY", publish["run"])
    test.assertLess(
        changes["steps"].index(reuse), changes["steps"].index(publish)
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
    test.assertIn("scripts.test_ci_rebase_reuse", contract["run"])
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
    test.assertIn(
        "steps.rebase-reuse.outputs.rebase_contract_changed == 'true'",
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
    test.assertIn("HEAD_SHA", bootstrap["env"])
    test.assertIn("diff --name-only --no-renames", bootstrap["run"])
    test.assertIn("GIT_NO_REPLACE_OBJECTS=1", bootstrap["run"])
    test.assertIn("/usr/bin/git", bootstrap["run"])
    test.assertIn("/usr/bin/python3", bootstrap["run"])
    test.assertNotIn("candidate-changed-paths.txt", bootstrap["run"])
    detect = next(
        step for step in changes["steps"] if step.get("id") == "detect"
    )
    test.assertLess(
        changes["steps"].index(bootstrap), changes["steps"].index(detect)
    )
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
    test.assertIn('[ "$rebase_contract" = "true" ]', record["run"])
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
    test.assertIn(
        "needs.changes.outputs.rebase_lane == 'full'",
        workflow["jobs"]["ci-fast"]["if"],
    )
    test.assertIn(
        "needs.changes.outputs.rebase_lane == 'ordinary'",
        workflow["jobs"]["ci-fast"]["if"],
    )
    for job_id in ("lint-rust", "script-unit"):
        test.assertIn(
            "needs.changes.outputs.rebase_lane != 'docs'",
            workflow["jobs"][job_id]["if"],
        )
        test.assertIn(
            "needs.changes.outputs.rebase_lane == 'targeted'",
            workflow["jobs"][job_id]["if"],
        )
    rust_policy = workflow["jobs"]["lint-rust"]
    nextest = next(
        step
        for step in rust_policy["steps"]
        if step.get("name") == "Install cargo-nextest for targeted Rust units"
    )
    test.assertEqual(
        nextest["if"],
        "needs.changes.outputs.rebase_lane == 'targeted'",
    )
    targeted_units = next(
        step
        for step in rust_policy["steps"]
        if step.get("name") == "Run Rust units for targeted rebase"
    )
    test.assertEqual(
        targeted_units["if"],
        "needs.changes.outputs.rebase_lane == 'targeted'",
    )
    test.assertEqual(
        targeted_units["run"],
        "python3 scripts/gate.py targeted-units",
    )
    test.assertFalse(targeted_units.get("continue-on-error", False))
    for job_id in (
        "diagnostic-kind-oracle",
        "backend-sanitizers",
        "smt-build",
        "smt-build-glibc231",
    ):
        test.assertIn(
            "needs.changes.outputs.rebase_lane != 'docs'",
            workflow["jobs"][job_id]["if"],
        )
    lint_summary = workflow["jobs"]["lint-and-unit"]
    test.assertIn("needs.changes.outputs.rebase_lane != 'docs'", lint_summary["if"])
    lint_gate = next(
        step
        for step in lint_summary["steps"]
        if step.get("name") == "Require every lint and unit worker"
    )
    test.assertNotIn("if", lint_gate)
    test.assertIn("lint-rust=", lint_gate["run"])
    test.assertIn("script-unit=", lint_gate["run"])

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
    test.assertIn("rebase_lane", changes["outputs"])
    test.assertIn("rebase_before", changes["outputs"])
    test.assertIn("rebase_contract_changed", changes["outputs"])
    test.assertEqual(
        changes["permissions"],
        {
            "actions": "read",
            "contents": "read",
            "issues": "read",
            "pull-requests": "read",
        },
    )
    test.assertEqual(
        changes["outputs"]["ci_contract_changed"],
        "${{ steps.candidate-preflight.outputs.ci_contract_changed }}",
    )
    test.assertIn("ci_validate_pr_candidate.py", str(changes))
    test.assertIn("ci_candidate_lifecycle.py", str(changes))
    test.assertIn("scripts.test_ci_candidate_lifecycle", str(changes))
    test.assertIn("scripts.test_ci_rebase_reuse", str(changes))
    test.assertIn(
        "../trusted-rebase/scripts/ci_rebase_reuse.py",
        str(changes),
    )
    bootstrap = next(
        step
        for step in changes["steps"]
        if step.get("id") == "ci-contract-bootstrap"
    )
    test.assertEqual(bootstrap["if"], "always()")
    for marker in BOOTSTRAP_CONTRACT_MARKERS:
        test.assertIn(marker, bootstrap["run"])
    test.assertIn("HEAD_SHA", bootstrap["env"])
    test.assertIn("diff --name-only --no-renames", bootstrap["run"])
    test.assertIn("GIT_NO_REPLACE_OBJECTS=1", bootstrap["run"])
    test.assertIn("/usr/bin/git", bootstrap["run"])
    test.assertIn("/usr/bin/python3", bootstrap["run"])
    test.assertNotIn("candidate-changed-paths.txt", bootstrap["run"])
    detect = next(
        step for step in changes["steps"] if step.get("id") == "detect"
    )
    test.assertLess(
        changes["steps"].index(bootstrap), changes["steps"].index(detect)
    )
    contract = next(
        step for step in changes["steps"] if step.get("id") == "ci-contract"
    )
    test.assertIn(
        "steps.ci-contract-bootstrap.outputs.ci_contract_changed == 'true'",
        contract["if"],
    )
    test.assertIn(
        "steps.rebase-reuse.outputs.rebase_contract_changed == 'true'",
        contract["if"],
    )
    record = next(
        step
        for step in changes["steps"]
        if step.get("id") == "candidate-preflight"
    )
    test.assertIn('[ "$rebase_contract" = "true" ]', record["run"])
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
    test.assertEqual(
        conformance["if"],
        "${{ !cancelled() && "
        "(needs.changes.outputs.candidate_preflight != 'success' || "
        "needs.changes.result != 'success' || "
        "needs.changes.outputs.rebase_lane == 'full' || "
        "(needs.changes.outputs.rebase_lane != 'docs' && "
        "(needs.changes.outputs.rebase_lane == 'targeted' || "
        "needs.changes.outputs.docs_only != 'true'))) }}",
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


def assert_author_machine_tokens(test: unittest.TestCase) -> None:
    for document in (AGENTS, AUTHOR_GUIDE):
        text = document.read_text()
        test.assertIn("Candidate-base-update:", text)
        test.assertIn("Candidate-history-rewrite:", text)
    guide = AUTHOR_GUIDE.read_text()
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
        assert_author_machine_tokens(self)

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

    def test_trusted_classifier_cannot_reuse_candidate_writable_paths(self) -> None:
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
            bootstrap["run"] += (
                '\npython3 "$classifier" < '
                '"$RUNNER_TEMP/candidate-changed-paths.txt"'
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

    def test_trusted_dispatch_is_full_while_ordinary_pr_updates_remain_docs_aware(
        self,
    ) -> None:
        for workflow_path in (CI, HULL):
            workflow = yaml.safe_load(workflow_path.read_text())
            with self.subTest(workflow=workflow_path.name, event="dispatch"):
                self.assertEqual(
                    run_rebase_selector(
                        workflow,
                        event_name="workflow_dispatch",
                        action="",
                    )["rebase_lane"],
                    "full",
                )
            with self.subTest(workflow=workflow_path.name, event="opened"):
                self.assertEqual(
                    run_rebase_selector(
                        workflow,
                        event_name="pull_request",
                        action="opened",
                    )["rebase_lane"],
                    "ordinary",
                )

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
