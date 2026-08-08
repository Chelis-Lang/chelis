"""Contract tests for shared CI action composition."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
DEVENV_YAML = ROOT / "devenv.yaml"
DEVENV_LOCK = ROOT / "devenv.lock"
CI_REVISION = "128d3acc50bb04bf75a6bb4cf34ec7f50dc388b9"
ACTIONLINT_VERSION = "1.7.12"
ACTIONLINT_SHA256 = (
    "sha256:8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8"
)
ZIZMOR_VERSION = "1.28.0"
ADOPTED_ACTIONS = frozenset(
    {
        "setup-devenv",
        "authenticate-private-ci-input",
        "reclaim-ubuntu-runner-disk",
        "nightly-status",
        "check-authorship",
        "openspec-governance",
        "actionlint",
        "zizmor",
        "prune-actions-cache",
    }
)
REMOTE_CI_ACTION = re.compile(
    r"Chelis-Lang/ci/actions/(?P<action>[a-z0-9-]+)@(?P<revision>[^\s#]+)"
)


def workflow_texts() -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8") for path in WORKFLOWS.glob("*.yml")
    }


def job_blocks(workflow: str) -> dict[str, str]:
    headers = list(re.finditer(r"(?m)^  (?P<name>[a-z0-9-]+):\s*$", workflow))
    return {
        header.group("name"): workflow[
            header.start() : headers[index + 1].start()
            if index + 1 < len(headers)
            else len(workflow)
        ]
        for index, header in enumerate(headers)
    }


def ci_input_revision(yaml_text: str) -> str:
    match = re.search(r"github:Chelis-Lang/ci/(?P<revision>[0-9a-f]{40})", yaml_text)
    if match is None:
        raise AssertionError("the ci Devenv input must use one full commit SHA")
    return match.group("revision")


def assert_pin_contract(
    yaml_text: str, lock_text: str, workflows: dict[str, str]
) -> None:
    configured = ci_input_revision(yaml_text)
    if configured != CI_REVISION:
        raise AssertionError("the configured ci revision is not the reviewed revision")
    locked = json.loads(lock_text)["nodes"]["ci"]["locked"]["rev"]
    if locked != configured:
        raise AssertionError(
            "the ci Devenv lock revision differs from the configured revision"
        )

    seen: set[str] = set()
    for name, text in workflows.items():
        for match in REMOTE_CI_ACTION.finditer(text):
            action = match.group("action")
            if action not in ADOPTED_ACTIONS:
                continue
            seen.add(action)
            if match.group("revision") != configured:
                raise AssertionError(f"{name}: {action} does not use the ci revision")
    if seen != ADOPTED_ACTIONS:
        raise AssertionError(f"the adopted action set differs: {sorted(seen)}")


def assert_private_auth_contract(workflows: dict[str, str]) -> None:
    setup = f"Chelis-Lang/ci/actions/setup-devenv@{CI_REVISION}"
    auth = f"Chelis-Lang/ci/actions/authenticate-private-ci-input@{CI_REVISION}"
    for workflow_name, text in workflows.items():
        for job_name, block in job_blocks(text).items():
            if setup not in block:
                continue
            project_devenv = re.search(r"(?m)^\s+run:.*\bdevenv\b", block)
            if project_devenv is None:
                continue
            setup_at = block.index(setup)
            auth_at = block.find(auth)
            if auth_at < 0:
                raise AssertionError(
                    f"{workflow_name}/{job_name}: missing private-ci authentication"
                )
            if not setup_at < auth_at < project_devenv.start():
                raise AssertionError(
                    f"{workflow_name}/{job_name}: invalid setup/authentication order"
                )
            private_suffix = block[auth_at:]
            if "app-client-id: ${{ vars.CI_APP_ID }}" not in private_suffix:
                raise AssertionError(
                    f"{workflow_name}/{job_name}: missing App client ID"
                )
            if (
                "app-private-key: ${{ secrets.CI_APP_PRIVATE_KEY }}"
                not in private_suffix
            ):
                raise AssertionError(
                    f"{workflow_name}/{job_name}: missing App private key"
                )
            if re.search(r"repositories:\s*ci\b", block):
                raise AssertionError(
                    f"{workflow_name}/{job_name}: retained inline ci token scope"
                )
            if "access-tokens = github.com=" in block:
                raise AssertionError(
                    f"{workflow_name}/{job_name}: retained inline Nix authentication"
                )


def assert_nightly_contract(workflows: dict[str, str]) -> None:
    expected = {
        "heavy-e2e.yml": (
            "needs: heavy-e2e",
            "result: ${{ needs.heavy-e2e.result }}",
            'issue-title: "Nightly failing: Heavy E2E"',
            "The **Heavy E2E** nightly failed.",
            "Nightly green again",
        ),
        "conformance-nightly.yml": (
            "needs: conformance-nightly",
            "result: ${{ needs.conformance-nightly.result }}",
            'issue-title: "Nightly failing: Hull Conformance"',
            "The **Hull Conformance** nightly failed.",
            "Nightly green again",
        ),
        "smt-full-prove.yml": (
            "needs: full-smt-prove",
            "result: ${{ needs.full-smt-prove.result }}",
            'issue-title: "Nightly failing: SMT Full Prove"',
            "The **SMT Full Prove** workflow failed.",
            "SMT Full Prove green again",
        ),
    }
    action = f"Chelis-Lang/ci/actions/nightly-status@{CI_REVISION}"
    for name, markers in expected.items():
        block = job_blocks(workflows[name])["report"]
        if action not in block:
            raise AssertionError(f"{name}: missing shared nightly status action")
        for marker in (*markers, "issue-label: nightly-failure"):
            if marker not in block:
                raise AssertionError(f"{name}: missing nightly marker {marker!r}")
        if "actions/github-script" in block:
            raise AssertionError(f"{name}: retained inline nightly reporter")


def assert_policy_contract(workflows: dict[str, str]) -> None:
    ci = job_blocks(workflows["ci.yml"])
    authorship = ci["no-ai-authorship"]
    for marker in (
        "name: No AI authorship markers",
        f"Chelis-Lang/ci/actions/check-authorship@{CI_REVISION}",
        "base-sha: ${{ steps.authorship-range.outputs.base-sha }}",
        "head-sha: ${{ steps.authorship-range.outputs.head-sha }}",
        "profile: all-markers",
        "max-commits: ${{ github.event.pull_request.commits || 2 }}",
    ):
        if marker not in authorship:
            raise AssertionError(f"authorship job missing {marker!r}")
    if "Co-authored-by: Claude" in authorship or "Generated-by:" in authorship:
        raise AssertionError("the inline authorship marker loop remains")

    lint = ci["lint-and-unit"]
    actionlint_markers = (
        f"Chelis-Lang/ci/actions/actionlint@{CI_REVISION}",
        f"actionlint-version: '{ACTIONLINT_VERSION}'",
        f"actionlint-sha256: {ACTIONLINT_SHA256}",
        "mode: enforce",
    )
    zizmor_markers = (
        f"Chelis-Lang/ci/actions/zizmor@{CI_REVISION}",
        f"zizmor-version: '{ZIZMOR_VERSION}'",
        "mode: advisory",
    )
    for marker in (*actionlint_markers, *zizmor_markers):
        if marker not in lint:
            raise AssertionError(f"lint-and-unit missing {marker!r}")

    openspec = workflows["openspec-validate.yml"]
    setup_at = openspec.find(f"Chelis-Lang/ci/actions/setup-devenv@{CI_REVISION}")
    governance_at = openspec.find(
        f"Chelis-Lang/ci/actions/openspec-governance@{CI_REVISION}"
    )
    if setup_at < 0 or governance_at < 0 or setup_at >= governance_at:
        raise AssertionError("OpenSpec setup must precede governance")
    if "mode: advisory" not in openspec:
        raise AssertionError("OpenSpec governance must remain advisory")


def assert_cache_contract(workflows: dict[str, str]) -> None:
    text = workflows["cache-prune.yml"]
    for marker in (
        'cron: "0 8 * * 1"',
        "actions: write",
        "pull-requests: read",
        "group: cache-prune",
        "cancel-in-progress: false",
        f"Chelis-Lang/ci/actions/setup-devenv@{CI_REVISION}",
        f"Chelis-Lang/ci/actions/prune-actions-cache@{CI_REVISION}",
        "github-token: ${{ github.token }}",
        "primary-ref: refs/heads/main",
        "keep-per-prefix: '1'",
        "protected-prefixes: cvc5-prebuilt-",
        "github.event_name == 'schedule'",
    ):
        if marker not in text:
            raise AssertionError(f"cache-prune workflow missing {marker!r}")
    if "scripts/ci_cache_prune.py" in text:
        raise AssertionError("the local cache-prune implementation remains")


class SharedCiCompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflows = workflow_texts()
        self.yaml = DEVENV_YAML.read_text(encoding="utf-8")
        self.lock = DEVENV_LOCK.read_text(encoding="utf-8")

    def test_reviewed_pin_and_action_set(self) -> None:
        assert_pin_contract(self.yaml, self.lock, self.workflows)

    def test_private_authentication_order_and_inputs(self) -> None:
        assert_private_auth_contract(self.workflows)

    def test_nightly_issue_contracts(self) -> None:
        assert_nightly_contract(self.workflows)

    def test_policy_action_contracts(self) -> None:
        assert_policy_contract(self.workflows)

    def test_cache_prune_contract(self) -> None:
        assert_cache_contract(self.workflows)

    def test_pin_mutations_fail_for_their_intended_reason(self) -> None:
        replacements = ("main", "v1", "128d3ac", "0" * 40)
        for replacement in replacements:
            mutated = self.workflows.copy()
            mutated["ci.yml"] = mutated["ci.yml"].replace(CI_REVISION, replacement, 1)
            with (
                self.subTest(replacement=replacement),
                self.assertRaisesRegex(AssertionError, "does not use the ci revision"),
            ):
                assert_pin_contract(self.yaml, self.lock, mutated)

    def test_authentication_mutations_fail_for_their_intended_reason(self) -> None:
        source = self.workflows["heavy-e2e.yml"]
        action = f"Chelis-Lang/ci/actions/authenticate-private-ci-input@{CI_REVISION}"
        mutations = {
            "missing-auth": source.replace(action, "missing-private-auth-action", 1),
            "inline-auth": source.replace(
                "          app-private-key: ${{ secrets.CI_APP_PRIVATE_KEY }}\n",
                "          app-private-key: ${{ secrets.CI_APP_PRIVATE_KEY }}\n"
                "        run: echo 'access-tokens = github.com=x'\n",
                1,
            ),
        }
        for name, mutation in mutations.items():
            changed = self.workflows.copy()
            changed["heavy-e2e.yml"] = mutation
            with self.subTest(name=name), self.assertRaises(AssertionError):
                assert_private_auth_contract(changed)

    def test_nightly_cache_and_policy_mutations_fail(self) -> None:
        mutations = (
            (assert_nightly_contract, "heavy-e2e.yml", "nightly-failure", "other"),
            (assert_policy_contract, "ci.yml", "mode: enforce", "mode: advisory"),
            (assert_cache_contract, "cache-prune.yml", "cvc5-prebuilt-", "other-"),
        )
        for validator, filename, old, new in mutations:
            changed = self.workflows.copy()
            changed[filename] = changed[filename].replace(old, new, 1)
            with self.subTest(filename=filename), self.assertRaises(AssertionError):
                validator(changed)


if __name__ == "__main__":
    unittest.main()
