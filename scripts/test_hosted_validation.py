"""Keep the CI coverage that replaces mandatory --validation execution live."""

import copy
from pathlib import Path
import unittest
from unittest import mock

import yaml

from scripts import gate
from scripts import ownership_ledger_tests as ledger_targets


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "python"
SKILL_COMMANDS = (
    f"{PYTHON} scripts/check_agent_skills.py",
    f"{PYTHON} -m unittest scripts.test_check_agent_skills scripts.test_hosted_validation",
    "cargo test -p chelis-conformance --test asset_drift_tripwire --test skill_set_uniformity",
)
MACOS_CENSUS_SELECTOR = (
    "binary_id(/^chelis-python::capacity_census_bindings$/) & "
    "test(/^registered_pyfunctions_match_the_reviewed_rustdoc_signatures$/)"
)
MACOS_WIRE_CENSUS_SELECTOR = (
    "binary_id(/^chelis-compiler-api::capacity_census_wire$/) & "
    "test(/^wire_schema_numeric_fields_match_the_reviewed_baseline$/)"
)
MACOS_WORKSPACE_SELECTOR = (
    f"not ({MACOS_CENSUS_SELECTOR}) & not ({MACOS_WIRE_CENSUS_SELECTOR})"
)
MACOS_WORKSPACE_COMMAND = (
    "cargo nextest run --workspace --profile ci-full "
    f"-E '{MACOS_WORKSPACE_SELECTOR}' "
    "--partition hash:${{ matrix.shard }}/3"
)
MACOS_CENSUS_COMMAND = (
    "cargo nextest run -p chelis-python --test capacity_census_bindings "
    f"--profile ci-full --no-tests=fail -E '{MACOS_CENSUS_SELECTOR}'"
)
MACOS_WIRE_CENSUS_COMMAND = (
    "cargo nextest run -p chelis-compiler-api --test capacity_census_wire "
    f"--profile ci-full --no-tests=fail -E '{MACOS_WIRE_CENSUS_SELECTOR}'"
)


def ledger_run(command):
    """The macOS spelling of a gate ledger command: its script under `python3`."""
    assert command[0] == gate.MANAGED_PYTHON, command
    return " ".join(["python3", *command[1:]])

def assert_hosted_coverage(test, workflow, nightly):
    jobs = workflow["jobs"]
    docs = jobs["docs"]
    test.assertEqual(docs.get("needs"), ["changes"])
    test.assertEqual(docs.get("if"), "${{ !cancelled() }}")
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
    test.assertEqual(macos["timeout-minutes"], 120)
    test.assertFalse(macos.get("continue-on-error", False))
    test.assertEqual(macos["strategy"]["matrix"]["shard"], [1, 2, 3])
    test.assertEqual(macos["name"], "macOS Workspace (shard ${{ matrix.shard }}/3)")
    workspace_steps = [
        step for step in macos["steps"]
        if "cargo nextest run --workspace" in step.get("run", "")
    ]
    test.assertEqual([step.get("run") for step in workspace_steps], [MACOS_WORKSPACE_COMMAND])
    test.assertNotIn("if", workspace_steps[0])
    test.assertFalse(workspace_steps[0].get("continue-on-error", False))
    test.assertIn("macos-binding-census", jobs)
    census = jobs["macos-binding-census"]
    test.assertEqual(census["runs-on"], "macos-latest")
    test.assertEqual(census["timeout-minutes"], 90)
    test.assertNotIn("if", census)
    test.assertFalse(census.get("continue-on-error", False))
    census_steps = [
        step for step in census["steps"] if "cargo nextest run" in step.get("run", "")
    ]
    test.assertEqual([step.get("run") for step in census_steps], [MACOS_CENSUS_COMMAND])
    test.assertNotIn("if", census_steps[0])
    test.assertFalse(census_steps[0].get("continue-on-error", False))
    test.assertEqual(
        census_steps[0].get("env", {}).get("CHELIS_CI_TIMING_DIR"),
        "${{ github.workspace }}/target/census-timings",
    )
    uploads = {
        step.get("with", {}).get("name"): step
        for step in census["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    }
    for name, path in (
        ("junit-macos-binding-census", "target/nextest/ci-full/junit.xml"),
        ("macos-binding-census-timing", "target/census-timings/"),
    ):
        test.assertIn(name, uploads)
        test.assertEqual(uploads[name].get("if"), "always()")
        test.assertEqual(uploads[name]["with"]["path"], path)
    test.assertIn("macos-wire-census", jobs)
    wire = jobs["macos-wire-census"]
    test.assertEqual(wire["runs-on"], "macos-latest")
    test.assertEqual(wire["timeout-minutes"], 90)
    test.assertNotIn("if", wire)
    test.assertFalse(wire.get("continue-on-error", False))
    wire_steps = [
        step for step in wire["steps"] if "cargo nextest run" in step.get("run", "")
    ]
    test.assertEqual([step.get("run") for step in wire_steps], [MACOS_WIRE_CENSUS_COMMAND])
    test.assertNotIn("if", wire_steps[0])
    test.assertFalse(wire_steps[0].get("continue-on-error", False))
    test.assertEqual(
        wire_steps[0].get("env", {}).get("CHELIS_CI_TIMING_DIR"),
        "${{ github.workspace }}/target/wire-census-timings",
    )
    wire_uploads = {
        step.get("with", {}).get("name"): step
        for step in wire["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    }
    for name, path in (
        ("junit-macos-wire-census", "target/nextest/ci-full/junit.xml"),
        ("macos-wire-census-timing", "target/wire-census-timings/"),
    ):
        test.assertIn(name, wire_uploads)
        test.assertEqual(wire_uploads[name].get("if"), "always()")
        test.assertEqual(wire_uploads[name]["with"]["path"], path)
    toolchains = [step for step in macos["steps"] if step.get("uses", "").startswith("dtolnay/rust-toolchain@")]
    test.assertEqual(len(toolchains), 1)
    test.assertIn("clippy", toolchains[0].get("with", {}).get("components", "").split(","))
    for command in (gate.CLIPPY_WORKSPACE, gate.CLIPPY_SOLVER_FREE_FEATURES):
        steps = [step for step in macos["steps"] if step.get("run") == " ".join(command)]
        test.assertEqual(len(steps), 1, command)
        test.assertEqual(steps[0].get("if"), "matrix.shard == 1")
        test.assertFalse(steps[0].get("continue-on-error", False))
    # The featureless workspace shards skip the ownership-ledger targets, so
    # their own job runs the gate's commands verbatim, the CLI command last
    # because it rebuilds ./target/debug/chelis. Each command derives its
    # targets from Cargo metadata; a hand-written cargo command would repeat
    # the list, so none may appear beside them.
    ledger = jobs["macos-ownership-ledger"]
    test.assertEqual(ledger["runs-on"], "macos-latest")
    test.assertNotIn("if", ledger)
    test.assertFalse(ledger.get("continue-on-error", False))
    commands = [
        step.get("run")
        for step in ledger["steps"]
        if step.get("run", "").startswith("cargo ")
        or "ownership_ledger_tests.py" in step.get("run", "")
    ]
    # Every ledger package, in run order, on both sides: a package the gate
    # runs and the workflow does not, or the reverse, fails here.
    gate_commands = [
        command
        for command in gate.STAGES["integration"]
        if command[1:2] == ["scripts/ownership_ledger_tests.py"]
    ]
    test.assertEqual(
        gate_commands,
        [
            [gate.MANAGED_PYTHON, "scripts/ownership_ledger_tests.py", package]
            for package in ledger_targets.LEDGER_PACKAGES
        ],
    )
    test.assertEqual(commands, [ledger_run(command) for command in gate_commands])
    for step in ledger["steps"]:
        test.assertNotIn("if", step)
        test.assertFalse(step.get("continue-on-error", False))
    aggregate = jobs["macos-smoke"]
    test.assertIn("macos-workspace-shard", aggregate["needs"])
    test.assertIn("macos-binding-census", aggregate["needs"])
    test.assertIn("macos-wire-census", aggregate["needs"])
    test.assertIn("macos-ownership-ledger", aggregate["needs"])
    test.assertEqual(aggregate.get("if"), "always()")
    test.assertFalse(aggregate.get("continue-on-error", False))
    command = (
        "python3 scripts/ci_require_success.py"
        " macos-workspace-shard=${{ needs.macos-workspace-shard.result }}"
        " macos-binding-census=${{ needs.macos-binding-census.result }}"
        " macos-wire-census=${{ needs.macos-wire-census.result }}"
        " macos-ownership-ledger=${{ needs.macos-ownership-ledger.result }}"
    )
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

    def test_macos_census_cannot_be_dropped_from_its_dedicated_job(self):
        for change in ("remove-job", "skip-job", "skip-step", "wrong-selector", "no-workspace-exclusion", "no-aggregate", "no-timing"):
            with self.subTest(change=change):
                workflow = copy.deepcopy(self.nightly)
                jobs = workflow["jobs"]
                if change == "remove-job":
                    del jobs["macos-binding-census"]
                elif change == "skip-job":
                    jobs["macos-binding-census"]["if"] = "false"
                elif change == "skip-step":
                    next(step for step in jobs["macos-binding-census"]["steps"] if "cargo nextest run" in step.get("run", ""))["if"] = "false"
                elif change == "wrong-selector":
                    next(step for step in jobs["macos-binding-census"]["steps"] if "cargo nextest run" in step.get("run", ""))["run"] = MACOS_CENSUS_COMMAND.replace("registered_pyfunctions", "unregistered_pyfunctions")
                elif change == "no-workspace-exclusion":
                    next(step for step in jobs["macos-workspace-shard"]["steps"] if "cargo nextest run --workspace" in step.get("run", ""))["run"] = "cargo nextest run --workspace --profile ci-full --partition hash:${{ matrix.shard }}/3"
                elif change == "no-aggregate":
                    jobs["macos-smoke"]["needs"].remove("macos-binding-census")
                else:
                    next(step for step in jobs["macos-binding-census"]["steps"] if step.get("with", {}).get("name") == "macos-binding-census-timing")["if"] = "false"
                with self.assertRaises(AssertionError):
                    assert_hosted_coverage(self, self.workflow, workflow)

    def test_macos_wire_census_cannot_be_dropped_from_its_dedicated_job(self):
        for change in ("remove-job", "skip-job", "skip-step", "wrong-selector", "no-workspace-exclusion", "no-aggregate", "no-timing"):
            with self.subTest(change=change):
                workflow = copy.deepcopy(self.nightly)
                jobs = workflow["jobs"]
                if change == "remove-job":
                    del jobs["macos-wire-census"]
                elif change == "skip-job":
                    jobs["macos-wire-census"]["if"] = "false"
                elif change == "skip-step":
                    next(step for step in jobs["macos-wire-census"]["steps"] if "cargo nextest run" in step.get("run", ""))["if"] = "false"
                elif change == "wrong-selector":
                    next(step for step in jobs["macos-wire-census"]["steps"] if "cargo nextest run" in step.get("run", ""))["run"] = MACOS_WIRE_CENSUS_COMMAND.replace("wire_schema_numeric_fields", "other_wire_schema_numeric_fields")
                elif change == "no-workspace-exclusion":
                    next(step for step in jobs["macos-workspace-shard"]["steps"] if "cargo nextest run --workspace" in step.get("run", ""))["run"] = MACOS_WORKSPACE_COMMAND.replace(
                        MACOS_WORKSPACE_SELECTOR, f"not ({MACOS_CENSUS_SELECTOR})"
                    )
                elif change == "no-aggregate":
                    jobs["macos-smoke"]["needs"].remove("macos-wire-census")
                else:
                    next(step for step in jobs["macos-wire-census"]["steps"] if step.get("with", {}).get("name") == "macos-wire-census-timing")["if"] = "false"
                with self.assertRaises(AssertionError):
                    assert_hosted_coverage(self, self.workflow, workflow)

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

    def test_ownership_ledger_must_run_on_macos_as_the_gate_spells_it(self):
        for command in (gate.OWNERSHIP_LEDGER_API_TESTS, gate.OWNERSHIP_LEDGER_CLI_TESTS):
            for change in ("missing", "skipped", "ignore-failure", "profile"):
                with self.subTest(command=command[-1], change=change):
                    workflow = copy.deepcopy(self.nightly)
                    steps = workflow["jobs"]["macos-ownership-ledger"]["steps"]
                    step = next(step for step in steps if step.get("run") == ledger_run(command))
                    if change == "missing":
                        steps.remove(step)
                    elif change == "skipped":
                        step["if"] = "false"
                    elif change == "ignore-failure":
                        step["continue-on-error"] = True
                    else:
                        step["run"] += " --profile ci-full"
                    with self.assertRaises(AssertionError):
                        assert_hosted_coverage(self, self.workflow, workflow)

    def test_a_hand_written_ledger_list_is_rejected(self):
        # The pre-derivation spelling repeated every target in the workflow.
        workflow = copy.deepcopy(self.nightly)
        steps = workflow["jobs"]["macos-ownership-ledger"]["steps"]
        step = next(step for step in steps if step.get("run") == ledger_run(gate.OWNERSHIP_LEDGER_CLI_TESTS))
        step["run"] = "cargo nextest run -p chelis-cli --features ownership-ledger --test issue_1314_json_bigint_ledger"
        with self.assertRaises(AssertionError):
            assert_hosted_coverage(self, self.workflow, workflow)

    def test_a_ledger_package_missing_from_either_side_is_rejected(self):
        packages = (*ledger_targets.LEDGER_PACKAGES, "chelis-ir")
        with mock.patch.object(ledger_targets, "LEDGER_PACKAGES", packages):
            with self.assertRaises(AssertionError):
                assert_hosted_coverage(self, self.workflow, self.nightly)
            # The workflow alone gaining the package still disagrees with the gate.
            workflow = copy.deepcopy(self.nightly)
            workflow["jobs"]["macos-ownership-ledger"]["steps"].append(
                {"name": "Ledger", "run": "python3 scripts/ownership_ledger_tests.py chelis-ir"}
            )
            with self.assertRaises(AssertionError):
                assert_hosted_coverage(self, self.workflow, workflow)

    def test_the_ledger_cli_command_runs_after_the_compiler_api_command(self):
        workflow = copy.deepcopy(self.nightly)
        steps = workflow["jobs"]["macos-ownership-ledger"]["steps"]
        api, cli = (
            next(index for index, step in enumerate(steps) if step.get("run") == ledger_run(command))
            for command in (gate.OWNERSHIP_LEDGER_API_TESTS, gate.OWNERSHIP_LEDGER_CLI_TESTS)
        )
        steps[api], steps[cli] = steps[cli], steps[api]
        with self.assertRaises(AssertionError):
            assert_hosted_coverage(self, self.workflow, workflow)

    def test_macos_producer_and_aggregate_cannot_be_skipped_or_made_nonblocking(self):
        for job_name in ("macos-workspace-shard", "macos-binding-census", "macos-wire-census", "macos-ownership-ledger", "macos-smoke"):
            for key, value in (("if", "false"), ("continue-on-error", True)):
                with self.subTest(job=job_name, key=key):
                    workflow = copy.deepcopy(self.nightly)
                    workflow["jobs"][job_name][key] = value
                    with self.assertRaises(AssertionError):
                        assert_hosted_coverage(self, self.workflow, workflow)

    def test_macos_aggregate_must_enforce_the_shard_result(self):
        for change in ("remove", "skip", "ignore-failure", "wrong-result", "wrong-ledger-result"):
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
                elif change == "wrong-result":
                    step["run"] = step["run"].replace("needs.macos-workspace-shard.result", "needs.changes.result")
                else:
                    step["run"] = step["run"].replace("needs.macos-ownership-ledger.result", "needs.changes.result")
                with self.assertRaises(AssertionError):
                    assert_hosted_coverage(self, self.workflow, workflow)


if __name__ == "__main__":
    unittest.main()
