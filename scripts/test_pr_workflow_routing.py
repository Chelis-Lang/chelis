"""Lock metadata-only CI routing and the manual expansion interface."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import yaml

from scripts import ci_candidate_receipt
from scripts import ci_pr_lifecycle_report


ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github/workflows/ci.yml"
ACKNOWLEDGEMENTS = ROOT / ".github/workflows/pr-contract-acknowledgements.yml"
EXPANSION = ROOT / ".github/workflows/pr-package-expansion.yml"
RETARGET = ROOT / ".github/workflows/pr-base-retarget.yml"
RECEIPT = ROOT / ".github/workflows/pr-candidate-receipt.yml"
CHANGELOG = ROOT / ".github/workflows/changelog.yml"
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
    "BASE_SHA",
}
RECEIPT_TRIGGER_WORKFLOWS = {
    "CI",
    "Hull Conformance",
    "Changelog",
    "PR Contract Acknowledgements",
    "PR Base Retarget Validation",
    "Secret scan",
}


def assert_candidate_cache_readonly(test: unittest.TestCase, workflow: dict) -> None:
    """Candidate jobs must inherit a server-enforced read-only cache token."""
    test.assertEqual(workflow.get("cache-mode"), "read")
    for name, job in workflow["jobs"].items():
        test.assertEqual(job.get("cache-mode", workflow["cache-mode"]), "read", name)


def assert_base_ref_is_shell_data(test: unittest.TestCase, workflow: dict) -> dict:
    steps = [step for job in workflow["jobs"].values() for step in job["steps"]]
    owners = [step for step in steps if step.get("name") in {
        "Record immutable candidate identity",
        "Dispatch CI and Hull and hold the head receipt",
    }]
    test.assertEqual(len(owners), 1)
    owner = owners[0]
    test.assertEqual(owner.get("env", {}).get("BASE_REF"),
                     "${{ github.event.pull_request.base.ref }}")
    test.assertIn('"$BASE_REF"', owner["run"])
    for step in steps:
        test.assertNotIn("github.event.pull_request.base.ref", step.get("run", ""))
    return owner

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


def assert_backstop_fetches_stay_connected(
    test: unittest.TestCase, body: str, targets: tuple[str, ...]
) -> None:
    """No backstop fetch of `targets` may carry a depth limit.

    `git fetch --depth=N` records the fetched commit in `.git/shallow`, and
    git then ignores the parents that commit records even when those parent
    objects are already in the clone. No later ordinary fetch removes the
    graft. Grafting the target snapshot cuts the one edge
    `ci_candidate_identity.py` crosses to reach the branch point, and its
    `git merge-base` then reports no common ancestor with an empty stderr
    (chelis#2228).
    """

    for target in targets:
        fetches = [
            line
            for line in body.splitlines()
            if "git fetch" in line and f"origin {target}" in line
        ]
        test.assertTrue(fetches, f"no backstop fetch of {target}")
        for line in fetches:
            test.assertNotIn(
                "--depth",
                line,
                f"the {target} backstop grafts the commit it fetches",
            )


def assert_candidate_deepening_stays_connected(
    test: unittest.TestCase, workflow: dict
) -> None:
    """The detector step reads an established clone; it no longer makes one.

    This used to assert the deepen and the two depth-less backstops inside
    the `detect` step, which is where chelis#2228's repair lived. That work
    moved into `Establish the candidate clone` and
    `scripts/ci_establish_candidate_clone.py`, which hold the whole
    property and more of it: the head deepen runs only against an already
    shallow clone, no other fetch in the job carries a depth argument, the
    first parent is read through a graft, and the result is asserted rather
    than assumed. `scripts/test_ci_establish_candidate_clone.py` owns
    those. What remains here is that the detector step delegates instead of
    fetching for itself, because a fetch reappearing here is exactly how
    the inherited-shape defect comes back.
    """

    steps = {
        step.get("id"): step for step in workflow["jobs"]["changes"]["steps"]
    }
    test.assertNotIn("git fetch", steps["detect"]["run"])
    owner = steps["clone"]["run"]
    test.assertIn("ci_establish_candidate_clone.py", owner)
    test.assertIn("--commits", owner)


def assert_never_cancels_in_progress(
    test: unittest.TestCase, scope: dict, label: str
) -> None:
    """No concurrency block governing these checks may cancel in progress.

    A bare string group is accepted: GitHub defaults `cancel-in-progress` to
    false for it. Anything else must say `false` literally, so neither a
    quoted string nor an expression can smuggle the cancellation back.
    """

    concurrency = scope.get("concurrency")
    if not isinstance(concurrency, dict):
        return
    # The key is matched case-insensitively as normalisation, not as a defence
    # against a known bypass: GitHub either rejects a workflow whose key it
    # does not recognise, which is loud, or ignores it and falls back to
    # false, which is harmless, so `Cancel-In-Progress: true` is a typo rather
    # than an evasion. Folding the case here is the same principle that makes
    # this guard reject a `${{ }}` expression in that field. A guard should
    # answer its own question rather than depend on how something it cannot
    # see is parsed.
    declared = [
        value
        for key, value in concurrency.items()
        if isinstance(key, str) and key.lower() == "cancel-in-progress"
    ]
    for value in declared or [False]:
        test.assertIs(
            value,
            False,
            f"{label} cancels a run already in progress",
        )


def assert_per_head_metadata_concurrency(
    test: unittest.TestCase, workflow: dict, prefix: str
) -> None:
    """A per-head check must not cancel its own in-flight run.

    Both of these checks are rerun by a pull-request body edit, and the agent
    contract asks for exactly that edit -- recording the reviewed head and its
    CI evidence -- immediately before merging. Cancelling the run already in
    flight for the same head leaves a `cancelled` check run that reads as a
    failure on a green pull request. Runs for different heads answer about
    different commits and never race, so keying the group by head SHA is what
    makes not cancelling safe as well as cheap.
    """

    concurrency = workflow["concurrency"]
    group = concurrency["group"]
    test.assertTrue(group.startswith(prefix), group)
    test.assertIn("github.event.pull_request.number", group)
    test.assertIn("github.event.pull_request.head.sha", group)
    assert_never_cancels_in_progress(test, workflow, "the workflow")
    # A job-level `concurrency` block does not replace the workflow-level one;
    # it adds a second group the job also belongs to, so `cancel-in-progress:
    # true` there restores exactly the cancellation this contract removes while
    # leaving the workflow-level block untouched. The idiom is in use here
    # (`pr-base-retarget.yml` gives its coordinator one), so reading only the
    # workflow level would leave the contract open to a plausible refactor.
    for job_id, job in workflow["jobs"].items():
        assert_never_cancels_in_progress(test, job, f"job {job_id}")


def assert_changelog_workflow(test: unittest.TestCase, workflow: dict) -> None:
    events = actions_events(workflow)
    # `edited` stays: GitHub delivers a base-branch change as an `edited`
    # event, and this check reads `base.sha`. A body-only or title-only edit
    # also arrives that way, which is wasteful but harmless; narrowing it with
    # a job-level `if` is not available, because a skipped required context
    # satisfies branch protection here and would let an edit turn a failing
    # Changelog green. `labeled`/`unlabeled` carry the `no-changelog` label.
    test.assertEqual(
        set(events["pull_request"]["types"]),
        {
            "opened",
            "synchronize",
            "reopened",
            "edited",
            "labeled",
            "unlabeled",
        },
    )
    test.assertEqual(workflow["permissions"], {"contents": "read"})
    job = workflow["jobs"]["changelog"]
    test.assertEqual(job["name"], "Changelog")
    text = str(job)
    test.assertIn("scripts.test_changelog", text)
    test.assertIn("changelog.py check-pr", text)
    assert_per_head_metadata_concurrency(test, workflow, "changelog-")


def assert_ci_metadata_routing(test: unittest.TestCase, workflow: dict) -> None:
    test.assertEqual(workflow["permissions"], {"contents": "read"})
    assert_candidate_deepening_stays_connected(test, workflow)
    authorship = next(
        step
        for step in workflow["jobs"]["no-ai-authorship"]["steps"]
        if step.get("name") == "Check commits for AI authorship markers"
    )
    # A grafted base stops excluding its own ancestors from `$BASE..$HEAD`,
    # so this scan would read commits that merged before the pull request.
    assert_backstop_fetches_stay_connected(
        test, authorship["run"], ('"$BASE"',)
    )
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
            "execution_target",
            "expected_candidate_sha",
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
    for output in (
        "rebase_packages",
        "rebase_run_rust",
        "rebase_run_script_unit",
        "rebase_run_integration",
        "rebase_run_smt",
        "rebase_run_backend",
        "rebase_run_diagnostic",
        "rebase_run_hull",
    ):
        test.assertIn(output, changes["outputs"])
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
    test.assertEqual(reuse["env"]["GH_TOKEN"], "${{ github.token }}")
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
    test.assertIn("scripts.test_gate_local.LocalCommandListTests", contract["run"])
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
    test.assertNotIn("candidate-changed-paths.txt", bootstrap["run"])
    detect = next(
        step for step in changes["steps"] if step.get("id") == "detect"
    )
    test.assertLess(
        changes["steps"].index(bootstrap), changes["steps"].index(detect)
    )
    # The verdict is split in two so the identity step can sit between
    # them: it needs a gate to run behind, and the verdict needs its
    # outcome. `scripts/test_ci_preflight_carrier.py` owns the always-
    # completes and single-carrier properties; what is checked here is
    # that the routing inputs still reach the pair.
    gate = next(
        step for step in changes["steps"] if step.get("id") == "preflight-gate"
    )
    record = next(
        step
        for step in changes["steps"]
        if step.get("id") == "candidate-preflight"
    )
    test.assertIn("candidate_preflight=", record["run"])
    test.assertIn("ci_contract_changed=", record["run"])
    test.assertIn('if [ "$lifecycle" != "success" ]', gate["run"])
    test.assertIn('if [ "$detected_contract" = "true" ]', gate["run"])
    test.assertIn('[ "$bootstrap_contract" = "true" ]', gate["run"])
    test.assertIn('[ "$rebase_contract" = "true" ]', gate["run"])
    identity = next(
        step
        for step in changes["steps"]
        if step.get("name") == "Record immutable candidate identity"
    )
    test.assertIn(
        "steps.preflight-gate.outputs.gate == 'success'", identity["if"]
    )
    test.assertIn("github.event_name == 'pull_request'", identity["if"])
    test.assertLess(
        changes["steps"].index(gate), changes["steps"].index(identity)
    )
    test.assertLess(
        changes["steps"].index(identity), changes["steps"].index(record)
    )
    test.assertEqual(identity["working-directory"], "candidate")
    test.assertIn("scripts/ci_candidate_identity.py", identity["run"])
    test.assertIn("--workflow-file ci.yml", identity["run"])
    test.assertNotIn("--base-sha", identity["run"])
    test.assertNotIn(
        "github.event.pull_request.base.sha", identity["run"]
    )
    upload = next(
        step
        for step in changes["steps"]
        if step.get("name") == "Upload immutable candidate identity"
    )
    # The upload now follows the identity it uploads rather than sharing
    # its condition, so an unrecorded identity is never published.
    test.assertIn("steps.identity.outcome == 'success'", upload["if"])
    test.assertEqual(upload["with"]["name"], "candidate-identity-ci")
    test.assertEqual(upload["with"]["if-no-files-found"], "error")
    test.assertIn("candidate-identity.json", upload["with"]["path"])
    test.assertIn('if [ "$contract_changed" = "true" ]', gate["run"])
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
    for job_id, flag in (
        ("lint-rust", "rebase_run_rust"),
        ("script-unit", "rebase_run_script_unit"),
    ):
        test.assertIn(
            "needs.changes.outputs.rebase_lane == 'targeted'",
            workflow["jobs"][job_id]["if"],
        )
        test.assertIn(flag, workflow["jobs"][job_id]["if"])
    rust_policy = workflow["jobs"]["lint-rust"]
    full_owner = next(
        step
        for step in rust_policy["steps"]
        if step.get("name") == "Gate (lint-and-unit subset)"
    )
    test.assertEqual(
        full_owner["if"],
        "needs.changes.outputs.rebase_lane != 'targeted' || needs.changes.outputs.rebase_packages == ''",
    )
    targeted_units = next(
        step
        for step in rust_policy["steps"]
        if step.get("name")
        == "Run affected Rust policy and units for targeted rebase"
    )
    test.assertEqual(
        targeted_units["if"],
        "needs.changes.outputs.rebase_lane == 'targeted' && needs.changes.outputs.rebase_packages != ''",
    )
    test.assertEqual(
        targeted_units["env"]["CHELIS_TARGETED_PACKAGES"],
        "${{ needs.changes.outputs.rebase_packages }}",
    )
    test.assertFalse(targeted_units.get("continue-on-error", False))
    for job_id, flag in (
        ("diagnostic-kind-oracle", "rebase_run_diagnostic"),
        ("backend-sanitizers", "rebase_run_backend"),
        ("smt-build", "rebase_run_smt"),
        ("smt-build-glibc231", "rebase_run_smt"),
    ):
        test.assertIn(flag, workflow["jobs"][job_id]["if"])
    lint_summary = workflow["jobs"]["lint-and-unit"]
    test.assertIn(
        "needs.changes.outputs.rebase_lane == 'targeted'", lint_summary["if"]
    )
    lint_gate = next(
        step
        for step in lint_summary["steps"]
        if step.get("name") == "Require the selected lint and unit frontier"
    )
    test.assertNotIn("if", lint_gate)
    test.assertIn("lint-rust=", lint_gate["run"])
    test.assertIn("script-unit=", lint_gate["run"])
    test.assertIn("RUN_RUST", lint_gate["env"])
    test.assertIn("RUN_SCRIPT_UNIT", lint_gate["env"])

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
    assert_candidate_deepening_stays_connected(test, workflow)
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
            "execution_target",
            "expected_candidate_sha",
        },
    )
    changes = workflow["jobs"]["changes"]
    test.assertIn("candidate_sha", changes["outputs"])
    test.assertIn("candidate_preflight", changes["outputs"])
    test.assertIn("rebase_lane", changes["outputs"])
    test.assertIn("rebase_before", changes["outputs"])
    test.assertIn("rebase_contract_changed", changes["outputs"])
    test.assertIn("rebase_run_hull", changes["outputs"])
    test.assertIn("rebase_packages", changes["outputs"])
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
    reuse = next(
        step for step in changes["steps"] if step.get("id") == "rebase-reuse"
    )
    test.assertEqual(reuse["env"]["GH_TOKEN"], "${{ github.token }}")
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
    gate = next(
        step for step in changes["steps"] if step.get("id") == "preflight-gate"
    )
    test.assertIn('[ "$rebase_contract" = "true" ]', gate["run"])
    identity = next(
        step
        for step in changes["steps"]
        if step.get("name") == "Record immutable candidate identity"
    )
    test.assertIn(
        "steps.preflight-gate.outputs.gate == 'success'", identity["if"]
    )
    test.assertIn("github.event_name == 'pull_request'", identity["if"])
    test.assertLess(
        changes["steps"].index(gate), changes["steps"].index(identity)
    )
    test.assertLess(
        changes["steps"].index(identity), changes["steps"].index(record)
    )
    test.assertEqual(identity["working-directory"], "candidate")
    test.assertIn("scripts/ci_candidate_identity.py", identity["run"])
    test.assertIn("--workflow-file conformance.yml", identity["run"])
    test.assertNotIn("--base-sha", identity["run"])
    test.assertNotIn(
        "github.event.pull_request.base.sha", identity["run"]
    )
    upload = next(
        step
        for step in changes["steps"]
        if step.get("name") == "Upload immutable candidate identity"
    )
    # The upload now follows the identity it uploads rather than sharing
    # its condition, so an unrecorded identity is never published.
    test.assertIn("steps.identity.outcome == 'success'", upload["if"])
    test.assertEqual(upload["with"]["name"], "candidate-identity-hull")
    test.assertEqual(upload["with"]["if-no-files-found"], "error")
    conformance = workflow["jobs"]["conformance"]
    # Hull skips a failed verdict, because the required Docs context
    # states the reason and Hull's red added nothing the two required
    # aggregators did not already say. It still runs when the `changes`
    # job itself did not succeed, which after this change is reachable
    # only through job setup, since every step in that job is tolerated. `scripts/test_ci_preflight_carrier.py` owns the
    # carrier property; this pins the exact expression.
    test.assertEqual(
        conformance["if"],
        "${{ !cancelled() && "
        "(needs.changes.result != 'success' || "
        "(needs.changes.outputs.candidate_preflight == 'success' && "
        "(needs.changes.outputs.rebase_lane == 'full' || "
        "(needs.changes.outputs.rebase_lane == 'targeted' && "
        "needs.changes.outputs.rebase_run_hull == 'true') || "
        "(needs.changes.outputs.rebase_lane == 'ordinary' && "
        "needs.changes.outputs.docs_only != 'true')))) }}",
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
    assert_per_head_metadata_concurrency(
        test, workflow, "pr-contract-acknowledgements-"
    )
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
    # A run triggered by another event (Secret scan's push runs, the retarget
    # coordinator's pull_request_target runs) skips its job; keying the group
    # by event keeps such a skipped run from cancelling a collection.
    test.assertIn(
        "github.event.workflow_run.event",
        str(workflow["concurrency"]["group"]),
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
    guide = AUTHOR_GUIDE.read_text()
    test.assertIn("Candidate-base-update:", guide)
    test.assertIn("Candidate-history-rewrite:", guide)
    test.assertIn("gh workflow run pr-package-expansion.yml", guide)
    test.assertIn("-f pr_number=", guide)
    test.assertIn("-f expected_head_sha=", guide)


class PullRequestWorkflowRoutingTests(unittest.TestCase):
    def test_preflight_requires_the_independent_local_command_contract(self) -> None:
        workflow = copy.deepcopy(yaml.safe_load(CI.read_text()))
        contract = next(
            step for step in workflow["jobs"]["changes"]["steps"]
            if step.get("id") == "ci-contract"
        )
        contract["run"] = contract["run"].replace(
            "scripts.test_gate_local.LocalCommandListTests", ""
        )
        with self.assertRaises(AssertionError):
            assert_ci_metadata_routing(self, workflow)

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
        assert_changelog_workflow(self, yaml.safe_load(CHANGELOG.read_text()))
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

    def test_a_fetch_reappearing_in_the_detector_step_is_rejected(self) -> None:
        for workflow_path, assertion in (
            (CI, assert_ci_metadata_routing),
            (HULL, assert_hull_retarget_dispatch),
        ):
            workflow = copy.deepcopy(yaml.safe_load(workflow_path.read_text()))
            detect = next(
                step
                for step in workflow["jobs"]["changes"]["steps"]
                if step.get("id") == "detect"
            )
            detect["run"] += '\ngit fetch --depth=1 origin "$BASE" || true\n'
            with self.subTest(workflow=workflow_path.name), self.assertRaises(
                AssertionError
            ):
                assertion(self, workflow)

    def test_the_detector_step_cannot_stop_delegating(self) -> None:
        for workflow_path, assertion in (
            (CI, assert_ci_metadata_routing),
            (HULL, assert_hull_retarget_dispatch),
        ):
            workflow = copy.deepcopy(yaml.safe_load(workflow_path.read_text()))
            owner = next(
                step
                for step in workflow["jobs"]["changes"]["steps"]
                if step.get("id") == "clone"
            )
            owner["run"] = owner["run"].replace(
                "ci_establish_candidate_clone.py", "true #", 1
            )
            with self.subTest(workflow=workflow_path.name), self.assertRaises(
                AssertionError
            ):
                assertion(self, workflow)

    def test_a_metadata_check_may_not_cancel_its_own_head(self) -> None:
        for workflow_path, assertion in (
            (CHANGELOG, assert_changelog_workflow),
            (ACKNOWLEDGEMENTS, assert_acknowledgement_workflow),
        ):
            for change in (
                "cancel",
                "group",
                "job-level",
                "quoted",
                "mixed-case",
            ):
                workflow = copy.deepcopy(
                    yaml.safe_load(workflow_path.read_text())
                )
                if change == "cancel":
                    workflow["concurrency"]["cancel-in-progress"] = True
                elif change == "quoted":
                    workflow["concurrency"]["cancel-in-progress"] = "true"
                elif change == "mixed-case":
                    del workflow["concurrency"]["cancel-in-progress"]
                    workflow["concurrency"]["Cancel-In-Progress"] = True
                elif change == "job-level":
                    job = next(iter(workflow["jobs"].values()))
                    job["concurrency"] = {
                        "group": "sneak-${{ github.event.pull_request.number }}",
                        "cancel-in-progress": True,
                    }
                else:
                    workflow["concurrency"]["group"] = workflow["concurrency"][
                        "group"
                    ].replace(
                        "-${{ github.event.pull_request.head.sha }}", ""
                    )
                with self.subTest(
                    workflow=workflow_path.name, change=change
                ), self.assertRaises(AssertionError):
                    assertion(self, workflow)

    def test_the_changelog_check_keeps_its_base_change_and_label_triggers(
        self,
    ) -> None:
        for dropped in ("edited", "labeled", "unlabeled"):
            workflow = copy.deepcopy(yaml.safe_load(CHANGELOG.read_text()))
            actions_events(workflow)["pull_request"]["types"].remove(dropped)
            with self.subTest(dropped=dropped), self.assertRaises(
                AssertionError
            ):
                assert_changelog_workflow(self, workflow)

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
        for workflow_path in (CI, HULL):
            with self.subTest(workflow=workflow_path.name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                candidate = root / "candidate"
                candidate.mkdir()
                environment = {
                    key: value for key, value in os.environ.items()
                    if not key.startswith("GIT_")
                }
                # Model the pre-activation host, not an enclosing developer shell.
                # Apple's /usr/bin shims otherwise delegate into Nix's SDK/PATH.
                for key in ("DEVELOPER_DIR", "SDKROOT", "TOOLCHAINS", "PYTHONPATH", "PYTHONHOME", "BASH_ENV", "ENV"):
                    environment.pop(key, None)
                environment.update(
                    GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                    GIT_AUTHOR_NAME="CI fixture", GIT_AUTHOR_EMAIL="ci@example.invalid",
                    GIT_COMMITTER_NAME="CI fixture", GIT_COMMITTER_EMAIL="ci@example.invalid",
                )

                def git(*arguments: str) -> str:
                    return subprocess.check_output(
                        ["git", *arguments], cwd=candidate, env=environment, text=True,
                        stderr=subprocess.PIPE,
                    ).strip()

                git("init", "-q")
                scripts = candidate / "scripts"
                scripts.mkdir()
                classifier = scripts / "ci_contract_paths.py"
                classifier.write_text((ROOT / "scripts/ci_contract_paths.py").read_text())
                (candidate / "README.md").write_text("base\n")
                git("add", ".")
                git("commit", "-qm", "base")
                base = git("rev-parse", "HEAD")
                (candidate / "README.md").write_text("documentation change\n")
                git("commit", "-qam", "docs")
                docs = git("rev-parse", "HEAD")
                (scripts / "gate.py").write_text("# changed contract\n")
                git("add", ".")
                git("commit", "-qm", "contract")
                head = git("rev-parse", "HEAD")

                # Neither the working-tree classifier nor PATH owns the decision.
                classifier.write_text("raise SystemExit(77)\n")
                poison = root / "poison"
                poison.mkdir()
                marker = root / "untrusted-executable-ran"
                for executable in ("git", "python3"):
                    shadow = poison / executable
                    shadow.write_text('#!/bin/sh\nprintf called > "$POISON_MARKER"\nexit 77\n')
                    shadow.chmod(0o755)
                output = root / "output"
                environment.update(
                    PATH=str(poison) + os.pathsep + environment["PATH"],
                    POISON_MARKER=str(marker), RUNNER_TEMP=str(root),
                    GITHUB_OUTPUT=str(output), EVENT_NAME="pull_request",
                    RUNNER_ENVIRONMENT=(
                        "self-hosted" if Path("/run/current-system/sw/bin/python3").is_file()
                        else "github-hosted"
                    ),
                )
                workflow = yaml.safe_load(workflow_path.read_text())
                body = next(
                    step["run"] for step in workflow["jobs"]["changes"]["steps"]
                    if step.get("id") == "ci-contract-bootstrap"
                )
                for before, after, expected in ((base, docs, "false"), (docs, head, "true")):
                    output.write_text("")
                    environment.update(BASE_SHA=before, HEAD_SHA=after)
                    result = subprocess.run(
                        ["bash", "-eu", "-o", "pipefail", "-c", body],
                        cwd=root, env=environment, text=True, capture_output=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(output.read_text().strip(), f"ci_contract_changed={expected}", result.stderr)
                    self.assertFalse(marker.exists(), "bootstrap executed a PATH shadow")

    def test_reviewed_dispatch_rejects_missing_or_moved_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = {
                key: value for key, value in os.environ.items()
                if not key.startswith("GIT_") and key not in {"BASH_ENV", "ENV"}
            }
            environment.update(
                GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                GIT_AUTHOR_NAME="CI fixture", GIT_AUTHOR_EMAIL="ci@example.invalid",
                GIT_COMMITTER_NAME="CI fixture", GIT_COMMITTER_EMAIL="ci@example.invalid",
            )

            def git(*arguments: str) -> str:
                return subprocess.check_output(
                    ["git", *arguments], cwd=root, env=environment, text=True,
                    stderr=subprocess.PIPE,
                ).strip()

            git("init", "-q")
            git("commit", "--allow-empty", "-qm", "reviewed candidate")
            reviewed = git("rev-parse", "HEAD")
            git("commit", "--allow-empty", "-qm", "moved candidate")
            current = git("rev-parse", "HEAD")
            output = root / "output"
            environment["GITHUB_OUTPUT"] = str(output)
            for workflow_path in (CI, HULL):
                workflow = yaml.safe_load(workflow_path.read_text())
                body = next(
                    step["run"] for step in workflow["jobs"]["changes"]["steps"]
                    if step.get("id") == "candidate"
                )
                for target, expected, accepted in (
                    ("", "", True),
                    ("self-hosted", current, True),
                    ("self-hosted", "", False),
                    ("self-hosted", reviewed, False),
                ):
                    with self.subTest(workflow=workflow_path.name, target=target, expected=expected):
                        output.write_text("")
                        environment.update(EXECUTION_TARGET=target, EXPECTED_CANDIDATE_SHA=expected)
                        result = subprocess.run(
                            ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", body],
                            cwd=root, env=environment, text=True, capture_output=True,
                        )
                        self.assertEqual(result.returncode == 0, accepted, result.stderr)
                        self.assertEqual(
                            output.read_text(),
                            f"candidate_sha={current}\n" if accepted else "",
                        )

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

    def test_receipt_triggers_are_the_collectors_mapped_workflows(self) -> None:
        mapped = {
            yaml.safe_load((ROOT / ".github/workflows" / name).read_text())["name"]
            for name in ci_candidate_receipt.WORKFLOWS
        }
        self.assertEqual(mapped, RECEIPT_TRIGGER_WORKFLOWS)
        trigger = actions_events(yaml.safe_load(RECEIPT.read_text()))
        self.assertEqual(set(trigger["workflow_run"]["workflows"]), mapped)
        self.assertEqual(
            ci_pr_lifecycle_report.WORKFLOW_RUN_PARENTS[
                ".github/workflows/pr-candidate-receipt.yml"
            ],
            mapped,
        )

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


class CandidatePrivilegeTests(unittest.TestCase):
    def test_candidate_cache_tokens_are_readonly(self) -> None:
        # Package expansion checks out the candidate from a main dispatch, so
        # it runs in the default branch's cache scope like a retarget.
        for path in (CI, HULL, EXPANSION):
            with self.subTest(workflow=path.name):
                assert_candidate_cache_readonly(self, yaml.safe_load(path.read_text()))

    def test_candidate_cache_write_grants_are_rejected(self) -> None:
        for path in (CI, HULL, EXPANSION):
            original = yaml.safe_load(path.read_text())
            for mode in (None, "write", "write-only"):
                workflow = copy.deepcopy(original)
                workflow["cache-mode"] = mode
                with self.subTest(workflow=path.name, mode=mode):
                    with self.assertRaises(AssertionError):
                        assert_candidate_cache_readonly(self, workflow)
            workflow = copy.deepcopy(original)
            workflow["jobs"][next(iter(workflow["jobs"]))]["cache-mode"] = "write"
            with self.assertRaises(AssertionError):
                assert_candidate_cache_readonly(self, workflow)

    def test_base_refs_are_passed_as_environment_data(self) -> None:
        for path in (CI, HULL, RETARGET):
            with self.subTest(workflow=path.name):
                assert_base_ref_is_shell_data(self, yaml.safe_load(path.read_text()))

    def test_interpolating_base_ref_in_shell_is_rejected(self) -> None:
        for path in (CI, HULL, RETARGET):
            workflow = yaml.safe_load(path.read_text())
            owner = assert_base_ref_is_shell_data(self, workflow)
            owner["run"] = owner["run"].replace(
                '"$BASE_REF"', '"${{ github.event.pull_request.base.ref }}"'
            )
            with self.assertRaises(AssertionError):
                assert_base_ref_is_shell_data(self, workflow)

    def test_git_valid_shell_active_refs_remain_literal_arguments(self) -> None:
        for path in (CI, HULL, RETARGET):
            owner = assert_base_ref_is_shell_data(self, yaml.safe_load(path.read_text()))
            for branch in ("release-$(touch${IFS}injected)",
                           "release-`touch${IFS}injected`", "release-normal"):
                with self.subTest(workflow=path.name, branch=branch):
                    subprocess.run(["git", "check-ref-format", "--branch", branch],
                                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    with tempfile.TemporaryDirectory() as directory:
                        temporary = Path(directory)
                        recorder = temporary / "record.py"
                        recorder.write_text(
                            "import json, pathlib, sys\n"
                            "pathlib.Path('arguments.json').write_text(json.dumps(sys.argv[1:]))\n"
                        )
                        run = re.sub(r"\$\{\{.*?\}\}", "safe-value", owner["run"])
                        # Preserve the workflow's argument quoting and execute its shell.
                        run = re.sub(r"^python3\s+scripts/\S+",
                                     f'"{sys.executable}" "{recorder}"', run)
                        environment = os.environ.copy()
                        environment.update(BASE_REF=branch, RUNNER_TEMP=directory)
                        subprocess.run(["bash", "-eu", "-c", run], cwd=directory,
                                       env=environment, check=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
                        arguments = json.loads((temporary / "arguments.json").read_text())
                        flag = "--expected-base-ref" if path == RETARGET else "--base-ref"
                        self.assertEqual(arguments[arguments.index(flag) + 1], branch)
                        self.assertFalse((temporary / "injected").exists())


if __name__ == "__main__":
    unittest.main()
