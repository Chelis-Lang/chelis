"""Contract tests for CI execution environment ownership."""

from __future__ import annotations

import hashlib
import re
import unittest
from enum import Enum
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
ACTIONS_DIR = ROOT / ".github" / "actions"
CI_REVISION = "a3b3e8ee939270649370826d62085c04342fd9e9"
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
    "actions/download-artifact@",
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
        "drift-source": JobDisposition.PROJECT_DEVENV,
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

# Project jobs can contain narrow portable or off-Nix probes. Every other
# direct command must enter the project through devenv-retry. The exact step
# names make each exception reviewable and make a new run step fail closed.
PROJECT_RUN_EXCEPTIONS: dict[tuple[str, str], dict[str, tuple[str, ...]]] = {
    ("ci.yml", "smt-build-darwin-arm64"): {
        "Assert Apple Silicon runner": ('test "$(uname -m)" = "arm64"',),
        "Verify packaged binary architecture": ("file target/release/chelis",),
    },
    ("ci.yml", "macos-smoke"): {
        "Assert Apple Silicon runner": ('test "$(uname -m)" = "arm64"',),
        "Verify packaged binary architecture": ("file target/debug/chelis",),
    },
    ("conformance-nightly.yml", "conformance-nightly"): {
        "Read pinned Hull commit": ("commit=$(python3 -c",),
    },
    ("ecosystem-drift.yml", "build-chelis"): {
        "Unpack the portable toolchain": (
            'tarball="$(find release-chelis',
            "tar -xzf",
        ),
        "Record HEAD chelis version": (
            'ver="$("$HEAD_TOOLCHAIN/bin/chelis" --version',
        ),
        "Stage toolchain tarball": ('staging="chelis-head-toolchain"',),
    },
    ("ecosystem-drift.yml", "drift-source"): {
        "Extract HEAD toolchain": ("tar -xzf toolchain/chelis-head-toolchain.tar.gz",),
        "Put HEAD chelis on PATH": (
            'test -n "$CHELIS_TOOLCHAIN"',
            'echo "$CHELIS_TOOLCHAIN/bin" >> "$GITHUB_PATH"',
        ),
        "File/update drift tracking issue": ("drift_report_issue.py",),
    },
    ("loc-report.yml", "loc-report"): {
        "Commit LOC report": (
            'git config user.name "github-actions[bot]"',
            'git commit -m "Update LOC report"',
        ),
    },
    ("nix-packages.yml", "nix-linux-x86-64"): {
        "Verify the runner system": (
            'subprocess.check_output(["nix", "eval", "--impure", "--raw", "--expr", "builtins.currentSystem"]',
            "x86_64-linux",
        ),
        "Verify the Nix sandbox": (
            'subprocess.check_output(["nix", "config", "show", "sandbox"]',
            'assert value == "true"',
        ),
        "Run the complete native flake check set": (
            "nix flake check --print-build-logs",
        ),
    },
    ("nix-packages.yml", "nix-darwin-arm64"): {
        "Verify the runner system": (
            'subprocess.check_output(["nix", "eval", "--impure", "--raw", "--expr", "builtins.currentSystem"]',
            "aarch64-darwin",
        ),
        "Verify the Nix sandbox": (
            'subprocess.check_output(["nix", "config", "show", "sandbox"]',
            'assert value == "true"',
        ),
        "Run the complete native flake check set": (
            "nix flake check --print-build-logs",
        ),
    },
    ("release.yml", "build-chelis-release"): {
        "Stage chelisup bootstrap script": (
            "cp crates/chelisup/bootstrap/chelisup.sh chelisup.sh",
        ),
    },
}

# The marker registry keeps each exception legible. The digest registry locks
# the complete step, so an added command cannot reuse an approved marker.
PROJECT_RUN_EXCEPTION_SHA256 = {
    ("ci.yml", "smt-build-darwin-arm64", "Assert Apple Silicon runner"): (
        "77669b1cf95f60d363dee7e392ec506fdfbbd59dbb14354efbf875aaf97688cb"
    ),
    ("ci.yml", "smt-build-darwin-arm64", "Verify packaged binary architecture"): (
        "097af4137b98dee109e77e6647de7f5a9416d701ed8fd520389d46fea2e9d04b"
    ),
    ("ci.yml", "macos-smoke", "Assert Apple Silicon runner"): (
        "7b61ebcd4922bc230abc36bb216278ce92b8bde11ca497b7f117614092fa9fcd"
    ),
    ("ci.yml", "macos-smoke", "Verify packaged binary architecture"): (
        "c40728a95770182dd5ed98fc4e3f9be4595cbe9b46f2fb0abc36403da3855a57"
    ),
    ("conformance-nightly.yml", "conformance-nightly", "Read pinned Hull commit"): (
        "6480e883b209480163613e86e4040421121a16b7d8a8c7d9e3c9447ffa54c386"
    ),
    ("ecosystem-drift.yml", "build-chelis", "Unpack the portable toolchain"): (
        "ac35503e7871e269df0cc927e7d64e55cec888e1b54be31a88ea5045cb4f4ede"
    ),
    ("ecosystem-drift.yml", "build-chelis", "Record HEAD chelis version"): (
        "7ac5961fce50ef198fa9fd3d77c95a3f51a6449e0e83256b88817e610342007f"
    ),
    ("ecosystem-drift.yml", "build-chelis", "Stage toolchain tarball"): (
        "0e368f900507278eee906d83901311962d21e1698a17cfa8c77e2b6f56f994fa"
    ),
    ("ecosystem-drift.yml", "drift-source", "Extract HEAD toolchain"): (
        "55fc03914e871f05757d89327c9c7a803ab8b933cac9f39c4310a071d95f6f1f"
    ),
    ("ecosystem-drift.yml", "drift-source", "Put HEAD chelis on PATH"): (
        "ce73161c9339c0d3e6887b8d73bbd52f521500fe87c198212b3bed5fc0250cf5"
    ),
    ("ecosystem-drift.yml", "drift-source", "File/update drift tracking issue"): (
        "c01c04a2d13792591452e935454af95e3bc8c2750007350bc8a8e22a7565c059"
    ),
    ("loc-report.yml", "loc-report", "Commit LOC report"): (
        "41b38a5031d8782259055b996c16faaa3fe73815b7f1fb9a07a626bc3a46bcd4"
    ),
    ("nix-packages.yml", "nix-linux-x86-64", "Verify the runner system"): (
        "b8987ef0fcf24c3fe9fee113a49f369207ee5926334a914609fa397f588177b0"
    ),
    ("nix-packages.yml", "nix-linux-x86-64", "Verify the Nix sandbox"): (
        "076a761b9106cfee4cc547ee3e82b308d3b218578423f787cad6942e26764a5d"
    ),
    (
        "nix-packages.yml",
        "nix-linux-x86-64",
        "Run the complete native flake check set",
    ): "bad2a5725d07497d76a3554278b7ebdebaa81eeaf2540035a8b2b29d4bceef54",
    ("nix-packages.yml", "nix-darwin-arm64", "Verify the runner system"): (
        "7b949b57c28bd525ba5f138d7a66ea22d0e40d58a3e4daf4cfb09ec293cd9d55"
    ),
    ("nix-packages.yml", "nix-darwin-arm64", "Verify the Nix sandbox"): (
        "076a761b9106cfee4cc547ee3e82b308d3b218578423f787cad6942e26764a5d"
    ),
    (
        "nix-packages.yml",
        "nix-darwin-arm64",
        "Run the complete native flake check set",
    ): "bad2a5725d07497d76a3554278b7ebdebaa81eeaf2540035a8b2b29d4bceef54",
    ("release.yml", "build-chelis-release", "Stage chelisup bootstrap script"): (
        "ef448eacb6304c8815433bef2e246e0662e350e9fc1556ec46cf2cf255bb266c"
    ),
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


def _run_step_blocks(job_block: str) -> list[tuple[str, str]]:
    """Return each direct run step with its display name."""
    lines = job_block.splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if line.startswith("      - ")]
    steps: list[tuple[str, str]] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        step = "".join(lines[start:end])
        if re.search(r"(?m)^        run:", step) is None and not lines[
            start
        ].startswith("      - run:"):
            continue
        name_match = re.match(r"      - name: (?P<name>.+?)\s*$", lines[start])
        name = name_match.group("name") if name_match is not None else "<unnamed>"
        steps.append((name, step))
    return steps


def _executes_devenv_retry(step: str) -> bool:
    executable = "\n".join(
        line for line in step.splitlines() if not line.lstrip().startswith("#")
    )
    command_patterns = (
        r"(?m)^\s*run:\s*(?:if\s+)?devenv-retry(?:\s|$)",
        r"(?m)^\s*(?:if\s+)?devenv-retry(?:\s|$)",
        r"\$\(\s*devenv-retry(?:\s|$)",
    )
    return any(
        re.search(pattern, executable) is not None for pattern in command_patterns
    )


def _contains_bare_project_command(step: str) -> bool:
    project_command = re.compile(
        r"(?:^|(?:&&|\|\||;)\s*)(?:if\s+!?\s*)?(?:exec\s+)?"
        r"(?:[^\s]+/)?(?:cargo|python3?|uv|chelis|mdbook)(?:\s|$)"
    )
    inside_devenv_script = False
    for raw_line in step.splitlines():
        command = raw_line.strip()
        if not command or command.startswith("#"):
            continue
        if command.startswith("run:"):
            command = command.removeprefix("run:").strip()
        if inside_devenv_script:
            if command == "'":
                inside_devenv_script = False
            continue
        bare_command = project_command.search(command) is not None
        if "devenv-retry" in command:
            if bare_command:
                return True
            if command.count("'") % 2 == 1:
                inside_devenv_script = True
            continue
        if bare_command:
            return True
    return inside_devenv_script


def _require_project_run_ownership(
    workflow_name: str, job_name: str, block: str, errors: list[str]
) -> None:
    exceptions = PROJECT_RUN_EXCEPTIONS.get((workflow_name, job_name), {})
    seen_steps: set[str] = set()
    for step_name, step in _run_step_blocks(block):
        seen_steps.add(step_name)
        if _executes_devenv_retry(step) and not _contains_bare_project_command(step):
            continue
        expected_markers = exceptions.get(step_name)
        exception_key = (workflow_name, job_name, step_name)
        expected_digest = PROJECT_RUN_EXCEPTION_SHA256.get(exception_key)
        actual_digest = hashlib.sha256(step.encode()).hexdigest()
        if expected_markers is None:
            errors.append(
                f"{workflow_name}/{job_name}/{step_name}: "
                "a project command bypasses devenv-retry"
            )
        elif (
            expected_digest is None
            or actual_digest != expected_digest
            or not all(marker in step for marker in expected_markers)
        ):
            errors.append(
                f"{workflow_name}/{job_name}/{step_name}: "
                "the registered project exception differs"
            )

    for missing_step in sorted(set(exceptions) - seen_steps):
        errors.append(
            f"{workflow_name}/{job_name}/{missing_step}: "
            "the registered project exception is absent"
        )


def execution_ownership_errors(
    workflows: dict[str, str], actions: dict[str, str]
) -> list[str]:
    """Return all errors in the closed CI execution ownership inventory."""
    errors: list[str] = []
    registered_exceptions = {
        (workflow_name, job_name, step_name)
        for (workflow_name, job_name), steps in PROJECT_RUN_EXCEPTIONS.items()
        for step_name in steps
    }
    if registered_exceptions != set(PROJECT_RUN_EXCEPTION_SHA256):
        errors.append("the project run exception digest inventory differs")
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

            if disposition in {
                JobDisposition.PROJECT_DEVENV,
                JobDisposition.PROJECT_WITH_PRE_SETUP_PROBE,
                JobDisposition.PROJECT_WITH_OFF_NIX_PROBE,
            }:
                if "devenv-retry" not in block:
                    errors.append(
                        f"{workflow_name}/{job_name}: missing a project Devenv command"
                    )
                _require_project_run_ownership(workflow_name, job_name, block, errors)

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

    def test_a_bare_project_command_fails(self) -> None:
        workflows = self.workflows.copy()
        block = job_blocks(workflows["loc-report.yml"])["loc-report"]
        mutated_block = block.replace(
            "      - name: Generate LOC report\n",
            "      - name: Bare project command\n"
            "        run: cargo --version\n\n"
            "      - name: Generate LOC report\n",
            1,
        )
        workflows["loc-report.yml"] = workflows["loc-report.yml"].replace(
            block, mutated_block, 1
        )
        errors = execution_ownership_errors(workflows, self.actions)
        self.assertIn(
            "loc-report.yml/loc-report/Bare project command: "
            "a project command bypasses devenv-retry",
            errors,
        )

    def test_a_devenv_retry_comment_does_not_own_a_project_command(self) -> None:
        workflows = self.workflows.copy()
        block = job_blocks(workflows["loc-report.yml"])["loc-report"]
        mutated_block = block.replace(
            "      - name: Generate LOC report\n",
            "      - name: Comment bypass\n"
            "        run: |\n"
            "          # devenv-retry does not execute here\n"
            "          cargo --version\n\n"
            "      - name: Generate LOC report\n",
            1,
        )
        workflows["loc-report.yml"] = workflows["loc-report.yml"].replace(
            block, mutated_block, 1
        )
        errors = execution_ownership_errors(workflows, self.actions)
        self.assertIn(
            "loc-report.yml/loc-report/Comment bypass: "
            "a project command bypasses devenv-retry",
            errors,
        )

    def test_a_mixed_project_step_rejects_a_bare_command(self) -> None:
        workflows = self.workflows.copy()
        block = job_blocks(workflows["loc-report.yml"])["loc-report"]
        mutated_block = block.replace(
            "      - name: Generate LOC report\n",
            "      - name: Mixed bypass\n"
            "        run: |\n"
            "          cargo --version\n"
            "          devenv-retry --profile ci shell --no-tui -- true\n\n"
            "      - name: Generate LOC report\n",
            1,
        )
        workflows["loc-report.yml"] = workflows["loc-report.yml"].replace(
            block, mutated_block, 1
        )
        errors = execution_ownership_errors(workflows, self.actions)
        self.assertIn(
            "loc-report.yml/loc-report/Mixed bypass: "
            "a project command bypasses devenv-retry",
            errors,
        )

    def test_a_same_line_command_after_devenv_retry_fails(self) -> None:
        workflows = self.workflows.copy()
        block = job_blocks(workflows["loc-report.yml"])["loc-report"]
        mutated_block = block.replace(
            "      - name: Generate LOC report\n",
            "      - name: Same-line bypass\n"
            "        run: devenv-retry --profile ci shell --no-tui -- true && cargo --version\n\n"
            "      - name: Generate LOC report\n",
            1,
        )
        workflows["loc-report.yml"] = workflows["loc-report.yml"].replace(
            block, mutated_block, 1
        )
        errors = execution_ownership_errors(workflows, self.actions)
        self.assertIn(
            "loc-report.yml/loc-report/Same-line bypass: "
            "a project command bypasses devenv-retry",
            errors,
        )

    def test_an_unrecognized_devenv_script_quote_fails_closed(self) -> None:
        workflows = self.workflows.copy()
        block = job_blocks(workflows["loc-report.yml"])["loc-report"]
        mutated_block = block.replace(
            "      - name: Generate LOC report\n",
            "      - name: Quote bypass\n"
            '        run: devenv-retry --profile ci shell --no-tui -- bash -c "echo it\'s fine"\n\n'
            "      - name: Generate LOC report\n",
            1,
        )
        workflows["loc-report.yml"] = workflows["loc-report.yml"].replace(
            block, mutated_block, 1
        )
        errors = execution_ownership_errors(workflows, self.actions)
        self.assertIn(
            "loc-report.yml/loc-report/Quote bypass: "
            "a project command bypasses devenv-retry",
            errors,
        )

    def test_a_registered_exception_rejects_another_command(self) -> None:
        workflows = self.workflows.copy()
        block = job_blocks(workflows["loc-report.yml"])["loc-report"]
        mutated_block = block.replace(
            '          git config user.name "github-actions[bot]"\n',
            "          cargo --version\n",
            1,
        )
        workflows["loc-report.yml"] = workflows["loc-report.yml"].replace(
            block, mutated_block, 1
        )
        errors = execution_ownership_errors(workflows, self.actions)
        self.assertIn(
            "loc-report.yml/loc-report/Commit LOC report: "
            "the registered project exception differs",
            errors,
        )

    def test_a_registered_exception_rejects_an_added_command(self) -> None:
        workflows = self.workflows.copy()
        block = job_blocks(workflows["loc-report.yml"])["loc-report"]
        mutated_block = block.replace(
            '          git config user.email "github-actions[bot]@users.noreply.github.com"\n',
            '          git config user.email "github-actions[bot]@users.noreply.github.com"\n'
            "          cargo --version\n",
            1,
        )
        workflows["loc-report.yml"] = workflows["loc-report.yml"].replace(
            block, mutated_block, 1
        )
        errors = execution_ownership_errors(workflows, self.actions)
        self.assertIn(
            "loc-report.yml/loc-report/Commit LOC report: "
            "the registered project exception differs",
            errors,
        )

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
