"""Contract tests for the push-first OpenSpec autoland controller.

The controller turns an ordinary branch push into an internal pull request,
without ever trusting anything the push carried.

The trust argument has one shape. A push runs a workflow FROM THE PUSHED
BRANCH, so that workflow is written by whoever pushed. It therefore gets
no permissions and produces no authorization: it exists only so that a
`workflow_run` event fires. Everything that matters -- which branch, which
commit, which repository, whether the diff is a document change -- the
controller re-derives from the API and from default-branch code. A rewritten
signal workflow can decline to run, which stops autoland, and that is the
only influence it has.

Tests below therefore never assert "GitHub would do X". Where GitHub's
behaviour matters -- specifically that a `GITHUB_TOKEN`-created pull request
does not start the required checks -- the assertion is made against this
repository's own workflow files, which is checkable evidence.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "openspec_controller.py"

SPEC = importlib.util.spec_from_file_location("openspec_controller", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load the controller module: {MODULE_PATH}")
controller = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = controller
SPEC.loader.exec_module(controller)

REPO = "Chelis-Lang/chelis"
BRANCH = "openspec/add-thing"
HEAD = "a" * 40
BASE = "b" * 40
RUN_ID = 12345
SUBMISSION_TOKEN = "ghp_submission_credential_value"
DEFAULT_TOKEN = "ghs_default_actions_token"

# `actions/create-github-app-token` v3, resolved to its commit. The repo's
# other App workflows use the floating `@v3` tag; this one pins the commit,
# because the credential it mints is the only write in the whole design.
APP_TOKEN_SHA = "bcd2ba49218906704ab6c1aa796996da409d3eb1"


class FakeApi:
    """Records every `gh` call and answers from a scripted world."""

    def __init__(
        self,
        *,
        run: dict | None = None,
        branch_sha: str | None = HEAD,
        base_sha: str = BASE,
        existing: list | None = None,
        verdict: int = 0,
        verdict_text: str = "verdict=auto\nreason=every changed path is an OpenSpec document",
        create_url: str = "https://github.com/Chelis-Lang/chelis/pull/77",
    ) -> None:
        self.run = run if run is not None else self.default_run()
        self.branch_sha = branch_sha
        self.base_sha = base_sha
        self.existing = existing if existing is not None else []
        self.verdict = verdict
        self.verdict_text = verdict_text
        self.create_url = create_url
        self.calls: list[list[str]] = []
        self.envs: list[dict] = []

    @staticmethod
    def default_run(**overrides: object) -> dict:
        payload = {
            "id": RUN_ID,
            "event": "push",
            "head_branch": BRANCH,
            "head_sha": HEAD,
            "path": ".github/workflows/openspec-autoland-signal.yml",
            "head_repository": {"full_name": REPO},
            "repository": {"full_name": REPO},
            "conclusion": "success",
        }
        payload.update(overrides)  # type: ignore[arg-type]
        return payload

    def __call__(self, command, **kwargs: object):
        self.calls.append(list(command))
        self.envs.append(dict(kwargs.get("env") or {}))
        joined = " ".join(command)
        if command[0] != "gh" and "openspec_acceptance.py" in joined:
            return _Result(self.verdict, self.verdict_text, "")
        if command[0] == "git":
            return _Result(0, "", "")
        target = command[-1] if command else ""
        if f"actions/runs/{RUN_ID}" in target:
            return _Result(0, json.dumps(self.run), "")
        if "/branches/" in target:
            name = target.split("/branches/", 1)[1]
            if name == BRANCH:
                if self.branch_sha is None:
                    return _Result(1, "", "gh: Not Found (HTTP 404)")
                return _Result(0, json.dumps({"commit": {"sha": self.branch_sha}}), "")
            return _Result(0, json.dumps({"commit": {"sha": self.base_sha}}), "")
        if command[:3] == ["gh", "pr", "list"]:
            return _Result(0, json.dumps(self.existing), "")
        if "--method" in command and "POST" in command:
            return _Result(
                0,
                json.dumps({"number": 77, "html_url": self.create_url}),
                "",
            )
        return _Result(0, "", "")

    def ran(self, needle: str) -> bool:
        return any(needle in " ".join(call) for call in self.calls)

    def create_call(self) -> tuple[list[str], dict] | None:
        for call, env in zip(self.calls, self.envs):
            if "--method" in call and "POST" in call:
                return call, env
        return None


class _Result:
    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run_controller(api: FakeApi, **kwargs: object) -> tuple[int, str]:
    out = io.StringIO()
    options = {
        "repository": REPO,
        "run_id": RUN_ID,
        "default_branch": "main",
        "repository_path": Path("."),
        "runner": api,
        "submission_token": SUBMISSION_TOKEN,
    }
    options.update(kwargs)
    with redirect_stdout(out), redirect_stderr(out):
        status = controller.run(**options)  # type: ignore[arg-type]
    return status, out.getvalue()


class HappyPathTests(unittest.TestCase):
    def test_a_document_push_opens_an_internal_pull_request(self) -> None:
        api = FakeApi()
        status, output = run_controller(api)
        self.assertEqual(status, controller.QUEUED, output)
        self.assertIsNotNone(api.create_call())
        self.assertTrue(api.ran("base=main"))
        self.assertIn("QUEUED", output)
        self.assertIn(api.create_url, output)

    def test_the_classification_uses_the_exact_pushed_commit(self) -> None:
        api = FakeApi()
        run_controller(api)
        classify = next(c for c in api.calls if "openspec_acceptance.py" in " ".join(c))
        self.assertIn(HEAD, classify)
        self.assertIn(BASE, classify)
        self.assertIn("--require-auto", classify)
        self.assertIn("--require-identical-governance", classify)

    def test_an_existing_open_pull_request_is_reused(self) -> None:
        api = FakeApi(
            existing=[
                {
                    "number": 77,
                    "state": "OPEN",
                    "url": "https://github.com/Chelis-Lang/chelis/pull/77",
                    "headRefOid": HEAD,
                }
            ]
        )
        status, output = run_controller(api)
        self.assertEqual(status, controller.QUEUED, output)
        self.assertIsNone(api.create_call())
        self.assertIn("reused", output.lower())

    def test_repeated_events_for_the_same_commit_do_not_duplicate(self) -> None:
        api = FakeApi()
        run_controller(api)
        creates = [c for c in api.calls if "--method" in c and "POST" in c]
        self.assertEqual(len(creates), 1)
        api.existing = [
            {"number": 77, "state": "OPEN", "url": api.create_url, "headRefOid": HEAD}
        ]
        run_controller(api)
        creates = [c for c in api.calls if "--method" in c and "POST" in c]
        self.assertEqual(len(creates), 1)


class RefusalTests(unittest.TestCase):
    def assert_blocked(self, api: FakeApi, needle: str) -> str:
        status, output = run_controller(api)
        self.assertIn(status, (controller.BLOCKED, controller.FAILED), output)
        self.assertIn(needle, output)
        self.assertIsNone(api.create_call(), "a write happened anyway")
        return output

    def test_a_mixed_push_writes_nothing(self) -> None:
        api = FakeApi(
            verdict=1,
            verdict_text=(
                "verdict=review\nreason=crates/chelis-types/src/lib.rs: "
                "outside the OpenSpec document boundary"
            ),
        )
        output = self.assert_blocked(api, "BLOCKED")
        self.assertIn("crates/chelis-types/src/lib.rs", output)

    def test_a_push_from_a_fork_is_refused(self) -> None:
        api = FakeApi(
            run=FakeApi.default_run(head_repository={"full_name": "someone/fork"})
        )
        self.assert_blocked(api, "not on")

    def test_a_non_push_event_is_refused(self) -> None:
        for event in ("pull_request", "workflow_dispatch", "schedule", "release"):
            with self.subTest(event=event):
                api = FakeApi(run=FakeApi.default_run(event=event))
                self.assert_blocked(api, "push")

    def test_the_default_branch_is_refused(self) -> None:
        api = FakeApi(run=FakeApi.default_run(head_branch="main"))
        self.assert_blocked(api, "default branch")

    def test_a_deleted_branch_is_refused(self) -> None:
        api = FakeApi(branch_sha=None)
        self.assert_blocked(api, "no longer")

    def test_a_branch_that_moved_since_the_push_is_refused(self) -> None:
        api = FakeApi(branch_sha="f" * 40)
        output = self.assert_blocked(api, "moved")
        self.assertIn(HEAD[:7], output)

    def test_a_malformed_branch_name_never_reaches_the_api(self) -> None:
        """Asserting the status is not enough: assert the name is absent.

        A refusal can happen for an unrelated reason further down, which
        would let a deleted name check keep the suite green while the value
        still reached three separate `gh` argv positions.
        """
        hostile = (
            "--upload-pack=touch /tmp/pwned",
            "a b",
            "",
            "-x",
            "refs\nheads/x",
            "a/../../../../users/octocat",
            "a..b",
            "trailing/",
            "x" * 400,
        )
        for name in hostile:
            with self.subTest(name=name):
                api = FakeApi(run=FakeApi.default_run(head_branch=name))
                status, _ = run_controller(api)
                self.assertIn(status, (controller.BLOCKED, controller.FAILED))
                self.assertIsNone(api.create_call())
                if name:
                    for call in api.calls:
                        self.assertNotIn(
                            name, " ".join(call), f"{name!r} reached {call}"
                        )

    def test_a_non_string_field_is_an_operational_failure_not_a_refusal(
        self,
    ) -> None:
        """A malformed payload must not exit with the refusal code."""
        for field, value in (
            ("head_repository", "not-a-dict"),
            ("head_repository", 7),
            ("head_branch", 1234),
        ):
            with self.subTest(field=field, value=value):
                api = FakeApi(run=FakeApi.default_run(**{field: value}))
                status, _ = run_controller(api)
                self.assertIn(status, (controller.BLOCKED, controller.FAILED))
                self.assertIsNone(api.create_call())

    def test_a_signal_run_from_another_workflow_file_is_refused(self) -> None:
        """The `workflows:` name filter is not provenance.

        A branch can add a second file whose `name:` matches, so the
        controller checks which FILE produced the run.
        """
        api = FakeApi(
            run=FakeApi.default_run(path=".github/workflows/evil-spoof.yml")
        )
        self.assert_blocked(api, "workflow")

    def test_a_malformed_head_sha_is_refused(self) -> None:
        api = FakeApi(run=FakeApi.default_run(head_sha="not-a-sha"))
        self.assert_blocked(api, "commit")

    def test_a_classifier_operational_failure_is_not_a_pass(self) -> None:
        api = FakeApi(verdict=2, verdict_text="openspec_acceptance: cannot run Git")
        status, output = run_controller(api)
        self.assertEqual(status, controller.FAILED)
        self.assertIsNone(api.create_call())
        self.assertIn("FAILED", output)


class UntrustedSignalTests(unittest.TestCase):
    """The signal run supplies an identity to look up, and nothing else."""

    def test_the_signal_runs_own_conclusion_is_not_trusted(self) -> None:
        api = FakeApi(run=FakeApi.default_run(conclusion="failure"))
        status, output = run_controller(api)
        self.assertEqual(status, controller.QUEUED, output)

    def test_no_artifact_is_ever_downloaded_or_read(self) -> None:
        api = FakeApi()
        run_controller(api)
        # Scoped to the endpoints and verbs that would fetch run output.
        # A bare "artifact" substring also matches the pull request body,
        # which is prose about OpenSpec artifacts, not a download.
        for forbidden in ("/artifacts", "download", "unzip", "actions/runs/12345/"):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(api.ran(forbidden))

    def test_no_code_from_the_pushed_branch_is_executed(self) -> None:
        api = FakeApi()
        run_controller(api)
        for call in api.calls:
            if "openspec_acceptance.py" in " ".join(call):
                script = next(part for part in call if part.endswith(".py"))
                with self.subTest(script=script):
                    self.assertFalse(script.startswith("head"))
        self.assertFalse(api.ran("git checkout"))
        self.assertFalse(api.ran("git merge"))

    def test_only_the_head_sha_and_branch_come_from_the_run(self) -> None:
        """A run payload cannot name a different repository to act on."""
        api = FakeApi(run=FakeApi.default_run(repository={"full_name": "other/repo"}))
        status, _ = run_controller(api)
        self.assertEqual(status, controller.QUEUED)
        for call in api.calls:
            with self.subTest(call=call):
                self.assertNotIn("other/repo", " ".join(call))


class EventSuppressionEvidenceTests(unittest.TestCase):
    """The required checks cannot run on a GITHUB_TOKEN-created request.

    This is asserted against this repository's real workflow files rather
    than against an assumption about GitHub. A pull request opened with
    `GITHUB_TOKEN` raises no `pull_request` event; if every required
    context is produced only by `pull_request`, then such a pull request can
    never satisfy branch protection, and the credential prerequisite the
    documentation states is real rather than defensive.
    """

    def workflow_triggers(self) -> dict[str, tuple[dict, dict]]:
        try:
            import yaml
        except ImportError:  # pragma: no cover - CI has PyYAML
            self.skipTest("PyYAML is unavailable")
        found: dict[str, tuple[dict, dict]] = {}
        for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
            document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            triggers = document.get(True) or document.get("on") or {}
            for job in (document.get("jobs") or {}).values():
                name = job.get("name")
                if name:
                    found[name] = (triggers, job)
        return found

    def test_at_least_one_required_context_needs_a_pull_request_event(
        self,
    ) -> None:
        """The measurement behind the credential prerequisite.

        On this branch `ci.yml` carries a workflow-level
        `workflow_dispatch`, so several required contexts COULD be produced
        without a pull request; on `origin/main` it does not, and none of
        them can. Either way `conformance.yml` has only `push: [main]` and
        `pull_request`, and `changelog.yml` on the default branch has only
        `pull_request`. One unreachable context is enough -- protection
        requires all nine -- so a `GITHUB_TOKEN`-opened pull request can
        never go green here.
        """
        jobs = self.workflow_triggers()
        unreachable = []
        found = 0
        for context in controller.REQUIRED_CONTEXTS_FOR_EVIDENCE:
            triggers, _ = jobs.get(context, ({}, {}))
            if not triggers:
                # Produced by a workflow this branch does not carry. Not
                # evidence either way, but counted so that deleting every
                # producer cannot leave this test vacuously green.
                continue
            found += 1
            other = set(triggers) - {"pull_request"}
            if "push" in other:
                branches = (triggers.get("push") or {}).get("branches") or []
                if branches == ["main"]:
                    other.discard("push")
            if not other:
                unreachable.append(context)
        self.assertGreaterEqual(
            found,
            4,
            "too few required-context producers were found to measure anything; "
            "the workflow set changed and this evidence test needs revisiting",
        )
        self.assertIn(
            "Hull Conformance Gate (Linux)",
            unreachable,
            "the conformance gate became reachable without a pull request; "
            "re-measure the prerequisite before relaxing the documentation",
        )

    def test_the_controller_documents_the_prerequisite(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("GITHUB_TOKEN", source)
        self.assertIn("does not", source)




class SignalWorkflowTests(unittest.TestCase):
    """The push-side workflow must stay powerless."""

    def setUp(self) -> None:
        self.path = REPO_ROOT / ".github" / "workflows" / "openspec-autoland-signal.yml"
        self.text = self.path.read_text(encoding="utf-8")

    def diagnostics(self, text: str) -> list[str]:
        found: list[str] = []
        import re as _re

        # Capability checks look at executable lines only: prose explaining
        # why the file holds nothing must not read as holding something
        # ("thou-gh-" contains "gh ").
        commands = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("#")
        )
        if not _re.search(r"(?m)^on:\n  push:", text):
            found.append("wrong-trigger")
        if _re.search(r"(?m)^\s+[a-z-]+:\s*(write|read)\s*$", text):
            found.append("permission-granted")
        if text.count("permissions: {}") != 2:
            found.append("permission-block")
        if "branches-ignore" not in text or "- main" not in text:
            found.append("default-branch-not-excluded")
        for forbidden in ("secrets.", "GH_TOKEN", "gh ", "uses:", "upload-artifact"):
            if forbidden in commands:
                found.append(f"capability:{forbidden.strip()}")
        # A newer push must supersede an in-flight signal run.
        if "cancel-in-progress: true" not in commands:
            found.append("concurrency-cancel")
        if "group: openspec-autoland-signal-${{ github.ref }}" not in commands:
            found.append("concurrency-group")
        return found

    def test_the_signal_workflow_holds_nothing(self) -> None:
        self.assertEqual(self.diagnostics(self.text), [])

    def test_each_property_can_fail(self) -> None:
        mutations = {
            "permission-granted": ("permissions: {}\n\nconcurrency", "permissions:\n  contents: write\n\nconcurrency"),
            "token-used": ("      - name: Record the push", "      - name: Leak\n        env:\n          GH_TOKEN: x\n        run: gh pr merge\n\n      - name: Record the push"),
            "action-used": ("      - name: Record the push", "      - uses: actions/checkout@v4\n\n      - name: Record the push"),
            "default-branch-included": ("    branches-ignore:\n      - main", "    branches:\n      - '**'"),
            "trigger-changed": ("on:\n  push:", "on:\n  pull_request_target:"),
            "supersession-dropped": ("cancel-in-progress: true", "cancel-in-progress: false"),
            "concurrency-rekeyed": (
                "group: openspec-autoland-signal-${{ github.ref }}",
                "group: openspec-autoland-signal-${{ github.run_id }}",
            ),
        }
        for name, (old, new) in mutations.items():
            with self.subTest(name=name):
                mutated = self.text.replace(old, new)
                self.assertNotEqual(mutated, self.text, "mutation changed nothing")
                self.assertTrue(self.diagnostics(mutated))


class ControllerWorkflowTests(unittest.TestCase):
    """The trusted side runs default-branch code and no head content."""

    def setUp(self) -> None:
        self.path = (
            REPO_ROOT / ".github" / "workflows" / "openspec-autoland-controller.yml"
        )
        self.text = self.path.read_text(encoding="utf-8")

    def diagnostics(self, text: str) -> list[str]:
        import re as _re

        found: list[str] = []
        commands = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("#")
        )
        if not _re.search(r"(?m)^on:\n  workflow_run:", text):
            found.append("untrusted-trigger")
        if 'workflows: ["OpenSpec autoland signal"]' not in text:
            found.append("signal-source")
        if "types: [completed]" not in text:
            found.append("trigger-types")
        if "github.event.workflow_run.event == 'push'" not in text:
            found.append("push-guard")
        if (
            "github.event.workflow_run.head_repository.full_name == github.repository"
            not in text
        ):
            found.append("same-repo-guard")
        if "ref: ${{ github.event.repository.default_branch }}" not in text:
            found.append("checkout-ref")
        if text.count("uses: actions/checkout@") != 1:
            found.append("checkout-count")
        if _re.search(r"(?m)^\s+ref: .*workflow_run\.head_branch", text):
            found.append("head-checkout")
        if "python3 base/scripts/openspec_controller.py" not in commands:
            found.append("controller-source")
        if "head/scripts" in text:
            found.append("code-from-head")
        # No merge, no approval, no auto-merge anywhere in this workflow.
        for forbidden in (
            "--auto",
            "--admin",
            "gh pr merge",
            "gh pr review",
            "contents: write",
            "artifact",
        ):
            if forbidden in commands:
                found.append(f"privileged-operation:{forbidden}")
        # The default token must be able to READ pull requests and nothing
        # more; the create call uses the submission credential.
        # Checked against executable lines: the header comment explains
        # each permission, and prose must not satisfy a permission check.
        if "pull-requests: read" not in commands:
            found.append("cannot-list-pull-requests")
        # A job-level permission entry, not the action's
        # `permission-pull-requests:` input, which contains the same text.
        if _re.search(r"(?m)^\s+pull-requests: write\s*$", commands):
            found.append("default-token-can-write-pull-requests")
        # `GET /actions/runs/{id}` is how the push identity is re-derived.
        if "actions: read" not in commands:
            found.append("no-actions-read")
        # The credential is a short-lived installation token minted from the
        # App this repository already configures, not a stored PAT.
        if "secrets.OPENSPEC_SUBMISSION_TOKEN" in commands:
            found.append("stored-pat-secret")
        if f"actions/create-github-app-token@{APP_TOKEN_SHA}" not in commands:
            found.append("app-token-action")
        if "app-id: ${{ vars.OPENSPEC_APP_ID }}" not in commands:
            found.append("app-id-source")
        if "private-key: ${{ secrets.OPENSPEC_APP_PRIVATE_KEY }}" not in commands:
            found.append("app-private-key-source")
        # The dedicated App only. The shared CI App's installation does not
        # grant pull-request write here -- minting from it returned HTTP 422,
        # "The permissions requested are not granted to this installation" --
        # and widening that App would hand pull-request write to every
        # workflow that already uses it for cross-repo reads.
        if "CI_APP_ID" in commands or "CI_APP_PRIVATE_KEY" in commands:
            found.append("shared-app-credentials")
        # Scoped to this repository only, never the whole installation.
        if "owner: ${{ github.repository_owner }}" not in commands:
            found.append("token-owner-scope")
        if "repositories: ${{ github.event.repository.name }}" not in commands:
            found.append("token-repository-scope")
        # Exactly one permission, and it is the one the create call needs.
        if "permission-pull-requests: write" not in commands:
            found.append("token-permission")
        if commands.count("permission-") != 1:
            found.append("token-extra-permissions")
        # Auto-revocation at job end is the action default; keeping the token
        # alive past the job would leave a live credential behind.
        if "skip-token-revoke" in commands:
            found.append("token-not-revoked")
        if (
            "OPENSPEC_SUBMISSION_TOKEN: ${{ steps.app-token.outputs.token }}"
            not in commands
        ):
            found.append("submission-token-env")
        if "must NOT be a required status check" not in text:
            found.append("required-check-warning")
        # One decision per branch, and a newer push supersedes an older one.
        if (
            "group: openspec-autoland-controller-${{ github.event.workflow_run.head_branch }}"
            not in commands
        ):
            found.append("concurrency-group")
        if "cancel-in-progress: true" not in commands:
            found.append("concurrency-cancel")
        if not _re.search(r"(?m)^\s+timeout-minutes: \d+", commands):
            found.append("no-timeout")
        # An ordinary refusal must not fail the run; an operational failure
        # must. Anything else hides one or reports the other as broken.
        if 'if [ "$status" -ge 2 ]; then exit "$status"; fi' not in commands:
            found.append("exit-status-policy")
        return found

    def test_the_controller_workflow_satisfies_its_contract(self) -> None:
        self.assertEqual(self.diagnostics(self.text), [])

    def test_each_property_can_fail(self) -> None:
        mutations = {
            "head-controlled-trigger": ("on:\n  workflow_run:", "on:\n  push:"),
            "push-guard-dropped": (
                "github.event.workflow_run.event == 'push'",
                "true",
            ),
            "same-repo-guard-dropped": (
                "github.event.workflow_run.head_repository.full_name == github.repository",
                "true",
            ),
            "head-checked-out": (
                "ref: ${{ github.event.repository.default_branch }}",
                "ref: ${{ github.event.workflow_run.head_branch }}",
            ),
            "controller-replaced": (
                "python3 base/scripts/openspec_controller.py",
                "python3 head/scripts/openspec_controller.py",
            ),
            "merge-added": (
                "      - name: Classify the push",
                '      - name: Merge\n        run: gh pr merge --auto\n\n      - name: Classify the push',
            ),
            "contents-write-granted": ("      contents: read", "      contents: write"),
            "signal-source-widened": (
                'workflows: ["OpenSpec autoland signal"]',
                'workflows: ["CI"]',
            ),
            "concurrency-rekeyed": (
                "group: openspec-autoland-controller-${{ github.event.workflow_run.head_branch }}",
                "group: openspec-autoland-controller-${{ github.event.workflow_run.id }}",
            ),
            "supersession-dropped": (
                "cancel-in-progress: true",
                "cancel-in-progress: false",
            ),
            "timeout-dropped": ("    timeout-minutes: 15\n", ""),
            "refusal-reported-as-failure": (
                'if [ "$status" -ge 2 ]; then exit "$status"; fi',
                'exit "$status"',
            ),
            "submission-token-removed": (
                "          OPENSPEC_SUBMISSION_TOKEN: ${{ steps.app-token.outputs.token }}\n",
                "",
            ),
            "submission-token-replaced-by-default-token": (
                "OPENSPEC_SUBMISSION_TOKEN: ${{ steps.app-token.outputs.token }}",
                "OPENSPEC_SUBMISSION_TOKEN: ${{ github.token }}",
            ),
            "stored-pat-reintroduced": (
                "OPENSPEC_SUBMISSION_TOKEN: ${{ steps.app-token.outputs.token }}",
                "OPENSPEC_SUBMISSION_TOKEN: ${{ secrets.OPENSPEC_SUBMISSION_TOKEN }}",
            ),
            "token-scope-widened-to-owner": (
                "          repositories: ${{ github.event.repository.name }}\n",
                "",
            ),
            "token-owner-scope-dropped": (
                "owner: ${{ github.repository_owner }}",
                "owner: Chelis-Lang",
            ),
            "token-permission-widened": (
                "permission-pull-requests: write",
                "permission-pull-requests: write\n          permission-contents: write",
            ),
            "token-revocation-disabled": (
                "          permission-pull-requests: write\n",
                "          permission-pull-requests: write\n          skip-token-revoke: true\n",
            ),
            "app-action-unpinned": (
                f"actions/create-github-app-token@{APP_TOKEN_SHA}",
                "actions/create-github-app-token@v3",
            ),
            "default-token-regains-write": (
                "      pull-requests: read",
                "      pull-requests: write",
            ),
            "actions-read-dropped": ("      actions: read\n", ""),
        }
        for name, (old, new) in mutations.items():
            with self.subTest(name=name):
                mutated = self.text.replace(old, new)
                self.assertNotEqual(mutated, self.text, "mutation changed nothing")
                self.assertTrue(self.diagnostics(mutated))

    def test_the_two_workflow_names_agree(self) -> None:
        signal = (
            REPO_ROOT / ".github" / "workflows" / "openspec-autoland-signal.yml"
        ).read_text(encoding="utf-8")
        name = signal.splitlines()[0].removeprefix("name:").strip()
        self.assertIn(f'workflows: ["{name}"]', self.text)




class SubmissionCredentialTests(unittest.TestCase):
    """The pull request is opened by a credential that is not GITHUB_TOKEN.

    That is not a stylistic choice. A pull request created with
    `GITHUB_TOKEN` raises no `pull_request` event, so the workflows that
    publish the required status checks never start and the pull request can
    never go green. The whole feature depends on the creating identity, so
    which token performs which call is asserted here rather than assumed.
    """

    def test_the_pull_request_is_created_with_the_submission_token(self) -> None:
        api = FakeApi()
        status, _ = run_controller(api)
        self.assertEqual(status, controller.QUEUED)
        call, env = api.create_call()
        self.assertEqual(env.get("GH_TOKEN"), SUBMISSION_TOKEN)
        self.assertIn(f"repos/{REPO}/pulls", " ".join(call))

    def test_the_default_token_never_creates_the_pull_request(self) -> None:
        api = FakeApi()
        run_controller(api, default_token=DEFAULT_TOKEN)
        _, env = api.create_call()
        self.assertNotEqual(env.get("GH_TOKEN"), DEFAULT_TOKEN)

    def test_read_calls_use_the_default_token(self) -> None:
        """The submission credential is for one write, and nothing else."""
        api = FakeApi()
        run_controller(api, default_token=DEFAULT_TOKEN)
        for call, env in zip(api.calls, api.envs):
            if "--method" in call and "POST" in call:
                continue
            if call and call[0] == "gh":
                with self.subTest(call=call):
                    self.assertNotEqual(
                        env.get("GH_TOKEN"),
                        SUBMISSION_TOKEN,
                        "a read used the submission credential",
                    )

    def test_a_missing_submission_token_blocks_before_any_write(self) -> None:
        api = FakeApi()
        status, output = run_controller(api, submission_token=None)
        self.assertEqual(status, controller.BLOCKED)
        self.assertIsNone(api.create_call())
        self.assertIn("BLOCKED", output)
        self.assertIn("OPENSPEC_SUBMISSION_TOKEN", output)

    def test_the_missing_token_message_names_the_exact_permission(self) -> None:
        api = FakeApi()
        _, output = run_controller(api, submission_token=None)
        self.assertIn("Pull requests", output)
        self.assertIn("write", output)

    def test_the_missing_token_message_points_at_the_configured_app(
        self,
    ) -> None:
        """The credential is a minted App token, not a stored PAT.

        The advice a maintainer reads must match how the workflow actually
        gets its credential, or it sends them to create a secret that
        nothing consumes.
        """
        api = FakeApi()
        _, output = run_controller(api, submission_token=None)
        self.assertIn("OPENSPEC_APP_ID", output)
        self.assertIn("OPENSPEC_APP_PRIVATE_KEY", output)
        self.assertIn("installation", output.lower())
        for stale in (
            "personal access token",
            "fine-grained",
            "repository secret OPENSPEC_SUBMISSION_TOKEN",
        ):
            with self.subTest(stale=stale):
                self.assertNotIn(stale, output)

    def test_the_advice_does_not_claim_the_installation_was_verified(
        self,
    ) -> None:
        """We can read that the App is configured, not what it may do."""
        source = MODULE_PATH.read_text(encoding="utf-8")
        for overclaim in ("installation is verified", "permission is verified"):
            with self.subTest(overclaim=overclaim):
                self.assertNotIn(overclaim, source)

    def test_a_blank_token_counts_as_missing(self) -> None:
        for value in ("", "   ", "\n"):
            with self.subTest(value=repr(value)):
                api = FakeApi()
                status, output = run_controller(api, submission_token=value)
                self.assertEqual(status, controller.BLOCKED)
                self.assertIsNone(api.create_call())
                self.assertIn("OPENSPEC_SUBMISSION_TOKEN", output)

    def test_reusing_an_existing_pull_request_needs_no_submission_token(
        self,
    ) -> None:
        """Nothing is written, so nothing needs the credential."""
        api = FakeApi(
            existing=[
                {
                    "number": 77,
                    "state": "OPEN",
                    "url": "https://github.com/Chelis-Lang/chelis/pull/77",
                    "headRefOid": HEAD,
                }
            ]
        )
        status, output = run_controller(api, submission_token=None)
        self.assertEqual(status, controller.QUEUED, output)
        self.assertIsNone(api.create_call())

    def test_a_missing_token_does_not_mask_an_ineligible_push(self) -> None:
        """A code push reports the boundary, not the credential.

        Checking the token first would report "missing secret" for every
        ordinary code push, which is noise about the wrong thing.
        """
        api = FakeApi(
            verdict=1,
            verdict_text=(
                "verdict=review\nreason=crates/chelis-types/src/lib.rs: "
                "outside the OpenSpec document boundary"
            ),
        )
        status, output = run_controller(api, submission_token=None)
        self.assertEqual(status, controller.BLOCKED)
        self.assertIn("crates/chelis-types/src/lib.rs", output)
        self.assertNotIn("OPENSPEC_SUBMISSION_TOKEN", output)

    def test_there_is_no_silent_fallback_to_the_default_token(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn('or os.environ.get("GITHUB_TOKEN")', source)
        self.assertNotIn('or os.environ.get("GH_TOKEN")', source)


class CredentialExposureTests(unittest.TestCase):
    """The submission credential exists only in the trusted workflow."""

    def workflow(self, name: str) -> str:
        return (REPO_ROOT / ".github" / "workflows" / name).read_text(
            encoding="utf-8"
        )

    def test_only_the_controller_workflow_references_the_app_key(self) -> None:
        secret = "OPENSPEC_APP_PRIVATE_KEY"
        controller_text = self.workflow("openspec-autoland-controller.yml")
        self.assertIn(secret, controller_text)
        for other in (
            "openspec-autoland-signal.yml",
            "openspec-autoland-validate.yml",
            "openspec-autoland.yml",
        ):
            with self.subTest(workflow=other):
                other_text = self.workflow(other)
                self.assertNotIn(secret, other_text)
                self.assertNotIn("OPENSPEC_APP_ID", other_text)
                self.assertNotIn("create-github-app-token", other_text)
                self.assertNotIn("OPENSPEC_SUBMISSION_TOKEN", other_text)

    def test_the_shared_ci_app_keeps_its_own_consumers(self) -> None:
        """This change must not disturb the App it did not switch.

        The dedicated App exists because the shared one lacks pull-request
        write here. The reverse must also hold: the workflows that mint
        cross-repo read tokens from the shared App keep doing exactly that,
        so a future reader cannot mistake this for a repository-wide
        migration off `CI_APP_ID`.
        """
        for name in ("conformance-nightly.yml", "ecosystem-drift.yml"):
            with self.subTest(workflow=name):
                text = self.workflow(name)
                self.assertIn("vars.CI_APP_ID", text)
                self.assertIn("secrets.CI_APP_PRIVATE_KEY", text)
                self.assertNotIn("OPENSPEC_APP_ID", text)
                self.assertNotIn("OPENSPEC_APP_PRIVATE_KEY", text)

    def test_no_head_controlled_workflow_can_reach_any_secret(self) -> None:
        """The signal and validator run head-supplied files."""
        for name in ("openspec-autoland-signal.yml", "openspec-autoland-validate.yml"):
            with self.subTest(workflow=name):
                self.assertNotIn("secrets.", self.workflow(name))

    def test_the_controller_passes_the_secret_to_exactly_one_step(self) -> None:
        text = self.workflow("openspec-autoland-controller.yml")
        # Executable lines only: the header comment names the secret while
        # explaining why it is there, and prose must not count as a use.
        commands = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("#")
        )
        self.assertEqual(commands.count("secrets.OPENSPEC_APP_PRIVATE_KEY"), 1)
        self.assertEqual(commands.count("steps.app-token.outputs.token"), 1)
        self.assertNotIn("secrets.OPENSPEC_SUBMISSION_TOKEN", commands)


if __name__ == "__main__":
    unittest.main()
