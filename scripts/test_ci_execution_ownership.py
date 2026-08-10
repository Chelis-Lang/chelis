"""Contract tests for CI execution environment ownership."""

from __future__ import annotations

import re
import unittest
from enum import Enum
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
ACTIONS_DIR = ROOT / ".github" / "actions"
CI_REVISION = "111b5865ccf04146344ad99dcdea9d724d65fc69"
SETUP_DEVENV = f"Chelis-Lang/ci/actions/setup-devenv@{CI_REVISION}"
AUTH_DEVENV = "Chelis-Lang/ci/actions/authenticate-private-ci-input@" + CI_REVISION
PORTABLE_SHELL = "devenv-ci bash --noprofile --norc -e -o pipefail {0}"
STEP_RUN = re.compile(r"(?m)^(?:      - run:|        run:)")
LOCAL_CVC5_ACTION = re.compile(r"uses: \./\.github/actions/cvc5-cache-(?:restore|save)")
MANAGED_ACTION_PREFIXES = (
    "./.github/actions/",
    "Chelis-Lang/ci/actions/",
    "Swatinem/rust-cache@",
    "actions/checkout@",
    "actions/create-github-app-token@",
    "actions/upload-artifact@",
)


class JobDisposition(Enum):
    """A closed execution disposition for one workflow job."""

    PROJECT_DEVENV = "project-devenv"
    PORTABLE_DEVENV = "portable-devenv"
    PROJECT_WITH_PRE_SETUP_PROBE = "project-with-pre-setup-probe"
    PROJECT_WITH_OFF_NIX_PROBE = "project-with-off-nix-probe"
    PRE_SETUP_ORCHESTRATION = "pre-setup-orchestration"
    OFF_NIX_VALIDATION = "off-nix-validation"
    SHARED_ACTIONS = "shared-actions"
    EXTERNAL_ACTIONS = "external-actions"


class ActionDisposition(Enum):
    """A closed execution disposition for one local composite action."""

    PORTABLE_DEVENV = "portable-devenv"
    PRE_SETUP_ORCHESTRATION = "pre-setup-orchestration"


JOB_DISPOSITIONS: dict[str, dict[str, JobDisposition]] = {
    "cache-prune.yml": {"prune": JobDisposition.SHARED_ACTIONS},
    "ci.yml": {
        "changes": JobDisposition.PRE_SETUP_ORCHESTRATION,
        "rejection-authority-liveness": JobDisposition.PROJECT_DEVENV,
        "diagnostic-kind-oracle": JobDisposition.PROJECT_DEVENV,
        "lint-and-unit": JobDisposition.PROJECT_DEVENV,
        "smt-build": JobDisposition.PROJECT_DEVENV,
        "smt-build-darwin-arm64": JobDisposition.PROJECT_WITH_PRE_SETUP_PROBE,
        "workspace-tests": JobDisposition.PROJECT_DEVENV,
        "dtype-phase3-oracle": JobDisposition.PROJECT_DEVENV,
        "faithful-observation-phase2-oracle": JobDisposition.PROJECT_DEVENV,
        "integration": JobDisposition.PORTABLE_DEVENV,
        "macos-smoke": JobDisposition.PROJECT_DEVENV,
        "backend-sanitizers": JobDisposition.PROJECT_DEVENV,
        "no-ai-authorship": JobDisposition.PORTABLE_DEVENV,
        "docs": JobDisposition.PROJECT_DEVENV,
    },
    "conformance-nightly.yml": {
        "conformance-nightly": JobDisposition.PROJECT_DEVENV,
        "report": JobDisposition.SHARED_ACTIONS,
    },
    "conformance.yml": {
        "changes": JobDisposition.PRE_SETUP_ORCHESTRATION,
        "conformance": JobDisposition.PROJECT_DEVENV,
    },
    "ecosystem-drift.yml": {
        "build-chelis": JobDisposition.PROJECT_WITH_OFF_NIX_PROBE,
        "drift": JobDisposition.OFF_NIX_VALIDATION,
    },
    "heavy-e2e.yml": {
        "heavy-e2e": JobDisposition.PROJECT_DEVENV,
        "report": JobDisposition.SHARED_ACTIONS,
    },
    "loc-report.yml": {"loc-report": JobDisposition.PROJECT_DEVENV},
    "nix-packages.yml": {
        "changes": JobDisposition.PRE_SETUP_ORCHESTRATION,
        "nix-linux-x86-64": JobDisposition.PROJECT_DEVENV,
        "nix-darwin-arm64": JobDisposition.PROJECT_DEVENV,
    },
    "openspec-validate.yml": {"openspec-validate": JobDisposition.SHARED_ACTIONS},
    "release.yml": {
        "build-chelis-release": JobDisposition.PROJECT_DEVENV,
        "consume-chelis-release": JobDisposition.OFF_NIX_VALIDATION,
        "consume-chelis-release-darwin": JobDisposition.OFF_NIX_VALIDATION,
        "build-chelisup-release": JobDisposition.PROJECT_DEVENV,
        "publish-release": JobDisposition.EXTERNAL_ACTIONS,
    },
    "smt-full-prove.yml": {
        "full-smt-prove": JobDisposition.PROJECT_DEVENV,
        "report": JobDisposition.SHARED_ACTIONS,
    },
}

ACTION_DISPOSITIONS = {
    "cvc5-cache-restore/action.yml": ActionDisposition.PORTABLE_DEVENV,
    "cvc5-cache-save/action.yml": ActionDisposition.PORTABLE_DEVENV,
    "detect-docs-only/action.yml": ActionDisposition.PRE_SETUP_ORCHESTRATION,
}


def workflow_texts() -> dict[str, str]:
    paths = [*WORKFLOWS_DIR.glob("*.yml"), *WORKFLOWS_DIR.glob("*.yaml")]
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(paths)}


def action_texts() -> dict[str, str]:
    paths = [
        *ACTIONS_DIR.rglob("action.yml"),
        *ACTIONS_DIR.rglob("action.yaml"),
    ]
    return {
        str(path.relative_to(ACTIONS_DIR)): path.read_text(encoding="utf-8")
        for path in sorted(paths)
    }


def job_blocks(workflow: str) -> dict[str, str]:
    """Parse top-level job blocks from one workflow."""
    jobs = re.search(r"(?m)^jobs:\s*$", workflow)
    if jobs is None:
        raise AssertionError("the workflow does not contain a jobs map")
    body = workflow[jobs.end() :]
    headers = list(re.finditer(r"(?m)^  (?P<name>[A-Za-z0-9_-]+):\s*$", body))
    return {
        header.group("name"): body[
            header.start() : headers[index + 1].start()
            if index + 1 < len(headers)
            else len(body)
        ]
        for index, header in enumerate(headers)
    }


def _require_setup_before_runs(
    workflow_name: str, job_name: str, block: str, errors: list[str]
) -> None:
    setup_at = block.find(SETUP_DEVENV)
    if setup_at < 0:
        errors.append(f"{workflow_name}/{job_name}: missing setup-devenv")
        return
    first_run = STEP_RUN.search(block)
    if first_run is not None and setup_at > first_run.start():
        errors.append(f"{workflow_name}/{job_name}: a run step precedes setup-devenv")


def _require_default_portable_shell(
    workflow_name: str, job_name: str, block: str, errors: list[str]
) -> None:
    if f"shell: {PORTABLE_SHELL}" not in block:
        errors.append(f"{workflow_name}/{job_name}: missing the portable Devenv shell")


def execution_ownership_errors(
    workflows: dict[str, str], actions: dict[str, str]
) -> list[str]:
    """Return all errors in the closed CI execution ownership inventory."""
    errors: list[str] = []
    if set(workflows) != set(JOB_DISPOSITIONS):
        errors.append("the workflow file inventory differs from JOB_DISPOSITIONS")

    for workflow_name, expected_jobs in JOB_DISPOSITIONS.items():
        text = workflows.get(workflow_name)
        if text is None:
            continue
        actual_jobs = job_blocks(text)
        if set(actual_jobs) != set(expected_jobs):
            errors.append(
                f"{workflow_name}: the job inventory differs from JOB_DISPOSITIONS"
            )
            continue

        for job_name, disposition in expected_jobs.items():
            block = actual_jobs[job_name]
            if disposition in {
                JobDisposition.PROJECT_DEVENV,
                JobDisposition.PORTABLE_DEVENV,
                JobDisposition.PROJECT_WITH_OFF_NIX_PROBE,
            }:
                _require_setup_before_runs(workflow_name, job_name, block, errors)
                _require_default_portable_shell(workflow_name, job_name, block, errors)

            if disposition is JobDisposition.PROJECT_WITH_PRE_SETUP_PROBE:
                setup_at = block.find(SETUP_DEVENV)
                if setup_at < 0:
                    errors.append(f"{workflow_name}/{job_name}: missing setup-devenv")
                _require_default_portable_shell(workflow_name, job_name, block, errors)
                probe_name = "name: Assert Apple Silicon runner"
                pre_setup_runs = sum(
                    run.start() < setup_at for run in STEP_RUN.finditer(block)
                )
                exact_probe = (
                    block.count("shell: bash") == 1
                    and block.find("shell: bash") < setup_at
                    and probe_name in block
                    and pre_setup_runs == 1
                )
                if not exact_probe:
                    errors.append(
                        f"{workflow_name}/{job_name}: the pre-setup probe differs"
                    )

            if (
                disposition
                in {
                    JobDisposition.PROJECT_DEVENV,
                    JobDisposition.PROJECT_WITH_PRE_SETUP_PROBE,
                    JobDisposition.PROJECT_WITH_OFF_NIX_PROBE,
                }
                and "devenv-retry" not in block
            ):
                errors.append(
                    f"{workflow_name}/{job_name}: missing a project Devenv command"
                )

            if (
                disposition
                in {
                    JobDisposition.PROJECT_DEVENV,
                    JobDisposition.PORTABLE_DEVENV,
                }
                and "shell: bash" in block
            ):
                errors.append(
                    f"{workflow_name}/{job_name}: an unowned host shell remains"
                )

            if disposition is JobDisposition.PROJECT_WITH_OFF_NIX_PROBE:
                probe_name = "name: Record HEAD chelis version"
                exact_probe = (
                    block.count("shell: bash") == 1
                    and block.find("shell: bash") > block.find(SETUP_DEVENV)
                    and probe_name in block
                )
                if not exact_probe:
                    errors.append(
                        f"{workflow_name}/{job_name}: the off-Nix probe differs"
                    )

            if disposition is JobDisposition.PRE_SETUP_ORCHESTRATION:
                if "uses: ./.github/actions/detect-docs-only" not in block:
                    errors.append(
                        f"{workflow_name}/{job_name}: missing docs-only detection"
                    )
                if SETUP_DEVENV in block:
                    errors.append(
                        f"{workflow_name}/{job_name}: docs-only detection must precede setup"
                    )
                if STEP_RUN.search(block) is not None:
                    errors.append(
                        f"{workflow_name}/{job_name}: docs-only orchestration runs local code"
                    )

            if (
                disposition is JobDisposition.OFF_NIX_VALIDATION
                and SETUP_DEVENV in block
            ):
                errors.append(
                    f"{workflow_name}/{job_name}: off-Nix validation uses Devenv"
                )

            if disposition is JobDisposition.SHARED_ACTIONS:
                if SETUP_DEVENV not in block:
                    errors.append(f"{workflow_name}/{job_name}: missing setup-devenv")
                if STEP_RUN.search(block) is not None:
                    errors.append(
                        f"{workflow_name}/{job_name}: a direct run step is not owned"
                    )

            if disposition is JobDisposition.EXTERNAL_ACTIONS:
                if SETUP_DEVENV in block or STEP_RUN.search(block) is not None:
                    errors.append(
                        f"{workflow_name}/{job_name}: external orchestration runs local code"
                    )

            managed_dispositions = {
                JobDisposition.PROJECT_DEVENV,
                JobDisposition.PORTABLE_DEVENV,
                JobDisposition.PROJECT_WITH_PRE_SETUP_PROBE,
                JobDisposition.PROJECT_WITH_OFF_NIX_PROBE,
                JobDisposition.PRE_SETUP_ORCHESTRATION,
                JobDisposition.SHARED_ACTIONS,
            }
            if disposition in managed_dispositions:
                for action in re.findall(r"(?m)^\s+(?:- )?uses: ([^\s#]+)", block):
                    if not action.startswith(MANAGED_ACTION_PREFIXES):
                        errors.append(
                            f"{workflow_name}/{job_name}: unclassified action {action}"
                        )

            for cvc5_call in LOCAL_CVC5_ACTION.finditer(block):
                setup_at = block.find(SETUP_DEVENV)
                auth_at = block.find(AUTH_DEVENV)
                invalid_order = (
                    setup_at < 0 or auth_at < setup_at or cvc5_call.start() < auth_at
                )
                if invalid_order:
                    errors.append(
                        f"{workflow_name}/{job_name}: cvc5 cache action precedes setup or authentication"
                    )

    if set(actions) != set(ACTION_DISPOSITIONS):
        errors.append("the local action inventory differs from ACTION_DISPOSITIONS")

    for action_name, disposition in ACTION_DISPOSITIONS.items():
        text = actions.get(action_name)
        if text is None:
            continue
        run_count = len(re.findall(r"(?m)^\s+run:", text))
        if disposition is ActionDisposition.PORTABLE_DEVENV:
            shell_count = text.count(f"shell: {PORTABLE_SHELL}")
            if run_count == 0 or shell_count != run_count:
                errors.append(
                    f"{action_name}: a run step bypasses the portable Devenv shell"
                )
            if re.search(r"(?m)^\s+shell: bash\s*$", text):
                errors.append(f"{action_name}: an unowned host shell remains")
        elif disposition is ActionDisposition.PRE_SETUP_ORCHESTRATION:
            required = "python3 scripts/ci_detect_docs_only.py"
            exact_host_step = run_count == 1 and text.count("shell: bash") == 1
            if (
                required not in text
                or f"shell: {PORTABLE_SHELL}" in text
                or not exact_host_step
            ):
                errors.append(f"{action_name}: the pre-setup exception differs")

    return errors


class CiExecutionOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflows = workflow_texts()
        self.actions = action_texts()

    def test_every_execution_surface_has_one_disposition(self) -> None:
        self.assertEqual(execution_ownership_errors(self.workflows, self.actions), [])

    def test_a_new_job_fails_closed(self) -> None:
        workflows = self.workflows.copy()
        workflows["ci.yml"] += "\n  new-host-job:\n    runs-on: ubuntu-latest\n"
        errors = execution_ownership_errors(workflows, self.actions)
        self.assertIn("ci.yml: the job inventory differs from JOB_DISPOSITIONS", errors)

    def test_a_project_host_shell_fails(self) -> None:
        workflows = self.workflows.copy()
        workflows["ci.yml"] = workflows["ci.yml"].replace(
            "      - name: Script unit tests (scripts/test_*.py)",
            "      - name: Host bypass\n        shell: bash\n        run: python3 -V\n\n"
            "      - name: Script unit tests (scripts/test_*.py)",
            1,
        )
        errors = execution_ownership_errors(workflows, self.actions)
        self.assertIn("ci.yml/lint-and-unit: an unowned host shell remains", errors)

    def test_a_host_tool_action_fails(self) -> None:
        workflows = self.workflows.copy()
        block = job_blocks(workflows["ci.yml"])["integration"]
        mutated_block = block.replace(
            "      - uses: actions/checkout@v6\n",
            "      - uses: actions/checkout@v6\n"
            "      - uses: actions/setup-python@v6\n",
            1,
        )
        workflows["ci.yml"] = workflows["ci.yml"].replace(block, mutated_block, 1)
        errors = execution_ownership_errors(workflows, self.actions)
        self.assertIn(
            "ci.yml/integration: unclassified action actions/setup-python@v6",
            errors,
        )

    def test_a_second_pre_setup_command_fails(self) -> None:
        workflows = self.workflows.copy()
        block = job_blocks(workflows["ci.yml"])["smt-build-darwin-arm64"]
        mutated_block = block.replace(
            "      - name: Set up portable Devenv\n",
            "      - run: uname -a\n\n      - name: Set up portable Devenv\n",
            1,
        )
        workflows["ci.yml"] = workflows["ci.yml"].replace(block, mutated_block, 1)
        errors = execution_ownership_errors(workflows, self.actions)
        self.assertIn(
            "ci.yml/smt-build-darwin-arm64: the pre-setup probe differs", errors
        )

    def test_a_run_before_setup_fails(self) -> None:
        workflows = self.workflows.copy()
        block = job_blocks(workflows["ci.yml"])["integration"]
        mutated_block = block.replace(
            "      - uses: actions/checkout@v6\n",
            "      - uses: actions/checkout@v6\n"
            "      - run: python3 scripts/ci_require_success.py sample=success\n",
            1,
        )
        workflows["ci.yml"] = workflows["ci.yml"].replace(block, mutated_block, 1)
        errors = execution_ownership_errors(workflows, self.actions)
        self.assertTrue(
            any("a run step precedes setup-devenv" in error for error in errors)
        )

    def test_a_cvc5_host_shell_fails(self) -> None:
        actions = self.actions.copy()
        actions["cvc5-cache-restore/action.yml"] = actions[
            "cvc5-cache-restore/action.yml"
        ].replace(f"shell: {PORTABLE_SHELL}", "shell: bash", 1)
        errors = execution_ownership_errors(self.workflows, actions)
        self.assertIn(
            "cvc5-cache-restore/action.yml: a run step bypasses the portable Devenv shell",
            errors,
        )

    def test_a_cvc5_call_before_authentication_fails(self) -> None:
        workflows = self.workflows.copy()
        block = job_blocks(workflows["ci.yml"])["smt-build"]
        auth_step = f"uses: {AUTH_DEVENV}"
        cache_step = "uses: ./.github/actions/cvc5-cache-restore"
        mutated_block = (
            block.replace(auth_step, "__AUTH_DEVENV__", 1)
            .replace(cache_step, auth_step, 1)
            .replace("__AUTH_DEVENV__", cache_step, 1)
        )
        workflows["ci.yml"] = workflows["ci.yml"].replace(block, mutated_block, 1)
        errors = execution_ownership_errors(workflows, self.actions)
        self.assertTrue(
            any(
                "cvc5 cache action precedes setup or authentication" in error
                for error in errors
            )
        )

    def test_the_docs_only_exception_rejects_a_portable_shell(self) -> None:
        actions = self.actions.copy()
        actions["detect-docs-only/action.yml"] = actions[
            "detect-docs-only/action.yml"
        ].replace("shell: bash", f"shell: {PORTABLE_SHELL}", 1)
        errors = execution_ownership_errors(self.workflows, actions)
        self.assertIn(
            "detect-docs-only/action.yml: the pre-setup exception differs", errors
        )

    def test_an_off_nix_job_rejects_devenv_setup(self) -> None:
        workflows = self.workflows.copy()
        block = job_blocks(workflows["release.yml"])["consume-chelis-release"]
        mutated_block = block.replace(
            "    steps:\n",
            f"    steps:\n      - uses: {SETUP_DEVENV}\n",
            1,
        )
        workflows["release.yml"] = workflows["release.yml"].replace(
            block, mutated_block, 1
        )
        errors = execution_ownership_errors(workflows, self.actions)
        self.assertIn(
            "release.yml/consume-chelis-release: off-Nix validation uses Devenv",
            errors,
        )

    def test_a_shared_action_job_rejects_a_direct_command(self) -> None:
        workflows = self.workflows.copy()
        block = job_blocks(workflows["cache-prune.yml"])["prune"]
        mutated_block = block.replace(
            "    steps:\n", "    steps:\n      - run: python3 -V\n", 1
        )
        workflows["cache-prune.yml"] = workflows["cache-prune.yml"].replace(
            block, mutated_block, 1
        )
        errors = execution_ownership_errors(workflows, self.actions)
        self.assertIn("cache-prune.yml/prune: a direct run step is not owned", errors)

    def test_an_external_action_job_rejects_a_direct_command(self) -> None:
        workflows = self.workflows.copy()
        block = job_blocks(workflows["release.yml"])["publish-release"]
        mutated_block = block.replace(
            "    steps:\n", "    steps:\n      - run: python3 -V\n", 1
        )
        workflows["release.yml"] = workflows["release.yml"].replace(
            block, mutated_block, 1
        )
        errors = execution_ownership_errors(workflows, self.actions)
        self.assertIn(
            "release.yml/publish-release: external orchestration runs local code",
            errors,
        )

    def test_a_new_local_action_fails_closed(self) -> None:
        actions = self.actions.copy()
        actions["new/action.yml"] = "runs:\n  using: composite\n"
        errors = execution_ownership_errors(self.workflows, actions)
        self.assertIn(
            "the local action inventory differs from ACTION_DISPOSITIONS", errors
        )


if __name__ == "__main__":
    unittest.main()
