"""Lock the required change-owned lane and isolated package-expansion trial."""

from __future__ import annotations

import copy
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/ci.yml"
EXPANSION_WORKFLOW = ROOT / ".github/workflows/pr-package-expansion.yml"
SHARDS = [0, 1, 2, 3]
PR_ONLY_JOB_IF = {
    "integration-plan": (
        "${{ !cancelled() && github.event_name != 'push' && "
        "needs.changes.outputs.candidate_preflight == 'success' && "
        "(needs.changes.result != 'success' || "
        "needs.changes.outputs.rebase_lane == 'full' || "
        "(needs.changes.outputs.rebase_lane == 'targeted' && "
        "needs.changes.outputs.rebase_run_integration == 'true') || "
        "(needs.changes.outputs.rebase_lane == 'ordinary' && "
        "needs.changes.outputs.docs_only != 'true')) }}"
    ),
    "change-owned-shard": (
        "${{ !cancelled() && github.event_name != 'push' && "
        "needs.integration-plan.result == 'success' }}"
    ),
    "change-owned-report": (
        "${{ always() && github.event_name != 'push' && "
        "(needs.changes.result != 'success' || "
        "needs.changes.outputs.rebase_lane == 'full' || "
        "(needs.changes.outputs.rebase_lane == 'targeted' && "
        "needs.changes.outputs.rebase_run_integration == 'true') || "
        "(needs.changes.outputs.rebase_lane == 'ordinary' && "
        "needs.changes.outputs.docs_only != 'true')) }}"
    ),
}


def _run_steps(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def _assert_pr_only_job(
    test: unittest.TestCase, name: str, job: dict
) -> None:
    test.assertEqual(
        " ".join(job["if"].split()),
        " ".join(PR_ONLY_JOB_IF[name].split()),
    )


def _assert_empty_shard_skips_preparation(
    test: unittest.TestCase, job: dict
) -> None:
    selection = next(
        step for step in job["steps"] if step.get("id") == "shard-selection"
    )
    test.assertIn("scripts/ci_change_owned.py prepare-shard", selection["run"])
    test.assertIn("${{ runner.temp }}", str(job))
    heavy_markers = (
        "free-disk-space",
        "ci_apt_get.py",
        "rust-toolchain",
        "setup-uv",
        "ci_setup_uv_python.py",
        "rust-cache",
        "uv pip install",
        "install-action@nextest",
    )
    for step in job["steps"]:
        if any(marker in str(step) for marker in heavy_markers):
            test.assertEqual(
                step.get("if"),
                "steps.shard-selection.outputs.has_targets == 'true'",
            )
    executor = next(
        step
        for step in job["steps"]
        if "scripts/ci_change_owned.py run-shard" in step.get("run", "")
    )
    test.assertEqual(
        executor.get("if"),
        "steps.shard-selection.outputs.has_targets == 'true'",
    )


def assert_change_owned_topology(
    test: unittest.TestCase, workflow: dict, expansion_workflow: dict
) -> None:
    jobs = workflow["jobs"]
    expected = {
        "integration-plan",
        "change-owned-shard",
        "change-owned-report",
    }
    test.assertLessEqual(expected, set(jobs))
    test.assertNotIn("package-expansion-shard", jobs)
    test.assertNotIn("package-expansion-summary", jobs)
    for name in expected:
        steps = jobs[name]["steps"]
        invoke = next(
            (
                index
                for index, step in enumerate(steps)
                if ".venv/bin/python scripts/ci_change_owned.py"
                in step.get("run", "")
            ),
            None,
        )
        test.assertIsNotNone(invoke, f"{name} requires managed Python invocation")
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
    test.assertIn("rebase_run_integration == 'true'", planner["if"])
    test.assertIn("needs.changes.result != 'success'", planner["if"])
    _assert_pr_only_job(test, "integration-plan", planner)
    checkout = next(
        step
        for step in planner["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )
    test.assertEqual(checkout.get("with", {}).get("fetch-depth"), 0)
    test.assertIn("scripts/ci_change_owned.py plan", _run_steps(planner))
    test.assertIn("targeted_rebase", _run_steps(planner))
    test.assertIn("REBASE_LANE", _run_steps(planner))
    test.assertIn("REBASE_BEFORE", _run_steps(planner))
    test.assertIn("REBASE_PACKAGES", _run_steps(planner))
    test.assertIn("--targeted-packages", _run_steps(planner))
    test.assertIn("integration-change-plan", str(planner))
    test.assertIn("target/integration-change/plan.json", str(planner))

    required = jobs["change-owned-shard"]
    test.assertEqual(required["needs"], ["changes", "integration-plan"])
    test.assertEqual(required["timeout-minutes"], 60)
    test.assertFalse(required.get("continue-on-error", False))
    test.assertFalse(required["strategy"]["fail-fast"])
    test.assertEqual(required["strategy"]["matrix"]["shard"], SHARDS)
    test.assertIn("needs.integration-plan.result == 'success'", required["if"])
    _assert_pr_only_job(test, "change-owned-shard", required)
    test.assertIn(
        "scripts/ci_change_owned.py run-shard", _run_steps(required)
    )
    test.assertIn("--lane change-owned", _run_steps(required))
    test.assertIn(
        "integration-change-owned-${{ matrix.shard }}", str(required)
    )
    _assert_empty_shard_skips_preparation(test, required)
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
        ["changes", "ci-fast", "integration-plan", "change-owned-shard"],
    )
    test.assertIn("always()", required_report["if"])
    _assert_pr_only_job(test, "change-owned-report", required_report)
    test.assertFalse(required_report.get("continue-on-error", False))
    required_report_commands = _run_steps(required_report)
    test.assertIn("scripts/ci_require_success.py", required_report_commands)
    test.assertIn(
        "change-owned-shard=${{ needs.change-owned-shard.result }}",
        required_report_commands,
    )
    test.assertIn("ci-fast-receipts", str(required_report))
    test.assertIn("--standing-coverage", required_report_commands)
    test.assertIn("REBASE_LANE", required_report_commands)
    test.assertIn(
        'if [ "$REBASE_LANE" != "targeted" ]',
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
    main_step = next(
        step
        for step in stable["steps"]
        if step.get("name") == "Require the fixed main integration suite"
    )
    test.assertEqual(main_step.get("if"), "github.event_name == 'push'")
    test.assertIn("ci-fast=${{ needs.ci-fast.result }}", main_step["run"])
    test.assertNotIn("change-owned-report", main_step["run"])
    pr_step = next(
        step
        for step in stable["steps"]
        if step.get("name")
        == "Require all ordinary pull-request integration legs"
    )
    test.assertIn("github.event_name != 'push'", pr_step.get("if"))
    test.assertIn(
        "needs.changes.outputs.rebase_lane != 'targeted'",
        pr_step.get("if"),
    )
    test.assertIn("ci-fast=${{ needs.ci-fast.result }}", pr_step["run"])
    test.assertIn(
        "change-owned-report=${{ needs.change-owned-report.result }}",
        pr_step["run"],
    )
    targeted_step = next(
        step
        for step in stable["steps"]
        if step.get("name") == "Require targeted rebase integration"
    )
    test.assertEqual(
        targeted_step.get("if"),
        "needs.changes.outputs.rebase_lane == 'targeted' && "
        "needs.changes.outputs.rebase_run_integration == 'true'",
    )
    test.assertNotIn("ci-fast=", targeted_step["run"])
    test.assertIn(
        "change-owned-report=${{ needs.change-owned-report.result }}",
        targeted_step["run"],
    )
    owner_only_step = next(
        step
        for step in stable["steps"]
        if step.get("name")
        == "Preserve integration context for an owner-only rebase"
    )
    test.assertIn(
        "rebase_run_integration != 'true'", owner_only_step.get("if")
    )
    test.assertNotIn("package-expansion", str(stable))

    expansion_jobs = expansion_workflow["jobs"]
    planner = expansion_jobs["integration-plan"]
    test.assertEqual(set(planner["outputs"]), {"base_ref", "candidate_sha"})
    expansion = expansion_jobs["package-expansion-shard"]
    test.assertEqual(
        expansion["needs"], ["integration-plan"]
    )
    test.assertEqual(expansion["timeout-minutes"], 20)
    test.assertFalse(expansion.get("continue-on-error", False))
    test.assertFalse(expansion["strategy"]["fail-fast"])
    test.assertEqual(expansion["strategy"]["matrix"]["shard"], SHARDS)
    expansion_commands = _run_steps(expansion)
    test.assertIn("scripts/ci_change_owned.py run-shard", expansion_commands)
    test.assertIn("--lane package-expansion", expansion_commands)
    test.assertIn(
        "integration-package-expansion-${{ matrix.shard }}", str(expansion)
    )
    _assert_empty_shard_skips_preparation(test, expansion)
    expansion_checkout = next(
        step
        for step in expansion["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )
    test.assertEqual(
        expansion_checkout["with"]["ref"],
        "${{ needs.integration-plan.outputs.candidate_sha }}",
    )

    summary = expansion_jobs["package-expansion-summary"]
    test.assertEqual(
        summary["needs"], ["integration-plan", "package-expansion-shard"]
    )
    test.assertIn("always()", summary["if"])
    test.assertFalse(summary.get("continue-on-error", False))
    summary_commands = _run_steps(summary)
    test.assertIn("scripts/ci_change_owned.py report", summary_commands)
    test.assertIn("--lane package-expansion", summary_commands)
    test.assertNotIn("--required", summary_commands)
    summary_checkouts = [
        step
        for step in summary["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    ]
    test.assertEqual(len(summary_checkouts), 2)
    trusted_checkout, candidate_checkout = summary_checkouts
    test.assertEqual(
        trusted_checkout.get("with", {}).get("ref"),
        "${{ github.sha }}",
    )
    test.assertEqual(trusted_checkout.get("with", {}).get("path"), "trusted")
    test.assertEqual(
        candidate_checkout.get("with", {}).get("ref"),
        "${{ needs.integration-plan.outputs.candidate_sha }}",
    )
    test.assertEqual(candidate_checkout.get("with", {}).get("path"), "candidate")
    test.assertEqual(candidate_checkout.get("with", {}).get("fetch-depth"), 0)
    summary_step = next(
        step
        for step in summary["steps"]
        if step.get("name") == "Summarize the informational package expansion"
    )
    test.assertEqual(summary_step.get("working-directory"), "candidate")
    test.assertIn(
        "python3 ../trusted/scripts/ci_validate_pr_candidate.py",
        summary_step["run"],
    )
    test.assertIn(
        ".venv/bin/python scripts/ci_change_owned.py report",
        summary_step["run"],
    )
    test.assertIn("--expected-base-ref", summary_step["run"])
    test.assertIn("--checkout-base-sha", summary_step["run"])
    test.assertNotIn("--expected-base-sha", summary_step["run"])
    test.assertIn("--validate-checkout", summary_step["run"])
    candidate_validation = summary_step["run"].split("report_status=0", 1)[0]
    test.assertIn("--plan target/integration-change/plan.json", candidate_validation)
    venv_step = next(
        step
        for step in summary["steps"]
        if step.get("name") == "Create uv-managed venv"
    )
    test.assertEqual(venv_step.get("working-directory"), "candidate")
    publish_step = next(
        step
        for step in summary["steps"]
        if step.get("name") == "Publish informational package-expansion summary"
    )
    test.assertEqual(publish_step.get("working-directory"), "candidate")
    upload_step = next(
        step
        for step in summary["steps"]
        if step.get("name") == "Upload informational package-expansion summary"
    )
    test.assertEqual(
        upload_step["with"]["path"],
        "candidate/target/integration-change/package-expansion-summary",
    )
    for step in summary["steps"]:
        if step.get("uses", "").startswith("actions/download-artifact@"):
            test.assertTrue(
                step["with"]["path"].startswith("candidate/target/integration-change")
            )


class ChangeOwnedWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = yaml.safe_load(WORKFLOW.read_text())
        self.expansion_workflow = yaml.safe_load(EXPANSION_WORKFLOW.read_text())

    def test_required_lane_and_informational_trial_are_isolated(self) -> None:
        assert_change_owned_topology(
            self, self.workflow, self.expansion_workflow
        )

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
                assert_change_owned_topology(
                    self, workflow, self.expansion_workflow
                )

    def test_expansion_cannot_feed_or_run_ahead_of_the_required_verdict(self) -> None:
        for mutation in ("feeds-stable", "moves-into-required-workflow"):
            workflow = copy.deepcopy(self.workflow)
            if mutation == "feeds-stable":
                workflow["jobs"]["integration"]["needs"].append(
                    "package-expansion-summary"
                )
            else:
                workflow["jobs"]["package-expansion-shard"] = copy.deepcopy(
                    self.expansion_workflow["jobs"]["package-expansion-shard"]
                )
            with self.subTest(mutation=mutation), self.assertRaises(
                AssertionError
            ):
                assert_change_owned_topology(
                    self, workflow, self.expansion_workflow
                )

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
                assert_change_owned_topology(
                    self, workflow, self.expansion_workflow
                )

    def test_main_push_cannot_enter_or_depend_on_the_change_owned_lane(self) -> None:
        for mutation in (
            "planner-on-push",
            "worker-on-push",
            "report-on-push",
            "planner-reenabled-on-push",
            "worker-reenabled-on-push",
            "report-reenabled-on-push",
            "main-requires-change-owned",
            "pr-omits-change-owned",
        ):
            workflow = copy.deepcopy(self.workflow)
            if mutation == "planner-on-push":
                job = workflow["jobs"]["integration-plan"]
                job["if"] = job["if"].replace(
                    "github.event_name != 'push' && ", ""
                )
            elif mutation == "worker-on-push":
                job = workflow["jobs"]["change-owned-shard"]
                job["if"] = job["if"].replace(
                    "github.event_name != 'push' && ", ""
                )
            elif mutation == "report-on-push":
                job = workflow["jobs"]["change-owned-report"]
                job["if"] = job["if"].replace(
                    "github.event_name != 'push' && ", ""
                )
            elif mutation.endswith("-reenabled-on-push"):
                job_name = {
                    "planner-reenabled-on-push": "integration-plan",
                    "worker-reenabled-on-push": "change-owned-shard",
                    "report-reenabled-on-push": "change-owned-report",
                }[mutation]
                job = workflow["jobs"][job_name]
                job["if"] = job["if"].replace(
                    " }}", " || github.event_name == 'push' }}"
                )
            elif mutation == "main-requires-change-owned":
                step = next(
                    step
                    for step in workflow["jobs"]["integration"]["steps"]
                    if step.get("name")
                    == "Require the fixed main integration suite"
                )
                step["run"] += (
                    "\n  change-owned-report="
                    "${{ needs.change-owned-report.result }}"
                )
            else:
                step = next(
                    step
                    for step in workflow["jobs"]["integration"]["steps"]
                    if step.get("name")
                    == "Require all ordinary pull-request integration legs"
                )
                step["run"] = step["run"].replace(
                    " change-owned-report="
                    "${{ needs.change-owned-report.result }}",
                    "",
                )
            with self.subTest(mutation=mutation), self.assertRaises(
                (AssertionError, StopIteration)
            ):
                assert_change_owned_topology(
                    self, workflow, self.expansion_workflow
                )

    def test_expansion_summary_cannot_mix_trusted_and_candidate_tools(self) -> None:
        for mutation in (
            "main-validator",
            "trusted-reporter",
            "unbound-checkout",
            "missing-candidate",
            "missing-venv-directory",
            "wrong-upload",
        ):
            expansion_workflow = copy.deepcopy(self.expansion_workflow)
            summary = expansion_workflow["jobs"]["package-expansion-summary"]
            if mutation == "main-validator":
                step = next(
                    step
                    for step in summary["steps"]
                    if step.get("name")
                    == "Summarize the informational package expansion"
                )
                step["run"] = step["run"].replace(
                    "python3 ../trusted/scripts/ci_validate_pr_candidate.py",
                    ".venv/bin/python scripts/ci_validate_pr_candidate.py",
                )
            elif mutation == "trusted-reporter":
                step = next(
                    step
                    for step in summary["steps"]
                    if step.get("name")
                    == "Summarize the informational package expansion"
                )
                step["run"] = step["run"].replace(
                    ".venv/bin/python scripts/ci_change_owned.py report",
                    "python3 ../trusted/scripts/ci_change_owned.py report",
                )
            elif mutation == "unbound-checkout":
                step = next(
                    step
                    for step in summary["steps"]
                    if step.get("name")
                    == "Summarize the informational package expansion"
                )
                step["run"] = step["run"].replace(
                    "--validate-checkout 2>&1",
                    "2>&1",
                )
            elif mutation == "missing-candidate":
                summary["steps"] = [
                    step
                    for step in summary["steps"]
                    if step.get("with", {}).get("path") != "candidate"
                ]
            elif mutation == "missing-venv-directory":
                step = next(
                    step
                    for step in summary["steps"]
                    if step.get("name") == "Create uv-managed venv"
                )
                step.pop("working-directory")
            else:
                step = next(
                    step
                    for step in summary["steps"]
                    if step.get("name")
                    == "Upload informational package-expansion summary"
                )
                step["with"]["path"] = (
                    "trusted/target/integration-change/package-expansion-summary"
                )
            with self.subTest(mutation=mutation), self.assertRaises(
                AssertionError
            ):
                assert_change_owned_topology(
                    self, self.workflow, expansion_workflow
                )

    def test_empty_shards_cannot_restore_caches_or_install_build_dependencies(self) -> None:
        for workflow, job_id in (
            (self.workflow, "change-owned-shard"),
            (self.expansion_workflow, "package-expansion-shard"),
        ):
            mutated = copy.deepcopy(workflow)
            job = mutated["jobs"][job_id]
            cache = next(
                step
                for step in job["steps"]
                if "rust-cache" in step.get("uses", "")
            )
            cache.pop("if")
            with self.subTest(job=job_id), self.assertRaises(AssertionError):
                assert_change_owned_topology(
                    self,
                    mutated if job_id == "change-owned-shard" else self.workflow,
                    mutated if job_id == "package-expansion-shard"
                    else self.expansion_workflow,
                )

    def test_empty_shards_cannot_enter_the_executor(self) -> None:
        for workflow, job_id in (
            (self.workflow, "change-owned-shard"),
            (self.expansion_workflow, "package-expansion-shard"),
        ):
            mutated = copy.deepcopy(workflow)
            executor = next(
                step
                for step in mutated["jobs"][job_id]["steps"]
                if "scripts/ci_change_owned.py run-shard"
                in step.get("run", "")
            )
            executor.pop("if")
            with self.subTest(job=job_id), self.assertRaises(AssertionError):
                assert_change_owned_topology(
                    self,
                    mutated if job_id == "change-owned-shard"
                    else self.workflow,
                    mutated if job_id == "package-expansion-shard"
                    else self.expansion_workflow,
                )


if __name__ == "__main__":
    unittest.main()
