"""Tests for the Cachix publisher of the Nix package closure.

Run with `.venv/bin/python scripts/test_publish_nix_cache.py`.
"""

from __future__ import annotations

import base64
import contextlib
import io
import re
import sys
import unittest
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import publish_nix_cache  # noqa: E402

SYSTEM = "aarch64-darwin"
CHELIS = "/nix/store/aaaa-chelis-0.19.4"
RUNTIME = "/nix/store/bbbb-chelis-runtime-0.19.4"
CHELISUP = "/nix/store/cccc-chelisup-0.19.4"
TOKEN = {publish_nix_cache.TOKEN_VARIABLE: "token"}
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
NIX_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nix-packages.yml"
PUBLISH_RUN = "run: python scripts/publish_nix_cache.py"
TAG_GATE = "if: startsWith(github.ref, 'refs/tags/v')"


class FakeNix:
    """Answers the three commands the publisher runs and records each call."""

    def __init__(self, fail: str | None = None) -> None:
        self.fail = fail
        self.calls: list[tuple[list[str], str | None]] = []

    def __call__(self, argv: Sequence[str], stdin: str | None) -> str:
        argv = list(argv)
        self.calls.append((argv, stdin))
        command = argv[1]
        if command == self.fail:
            raise publish_nix_cache.PublishError(f"{command} failed")
        if command == "eval":
            return f"{SYSTEM}\n"
        if command == "build":
            return f"{CHELIS}\n{RUNTIME}\n{CHELISUP}\n"
        if command == "run":
            return ""
        raise AssertionError(f"unexpected command {argv!r}")

    def pushes(self) -> list[tuple[list[str], str | None]]:
        return [call for call in self.calls if call[0][1] == "run"]


def run_main(
    nix: FakeNix, argv: Sequence[str] = (), environ: dict[str, str] = TOKEN
) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = publish_nix_cache.main(list(argv), run=nix, environ=environ)
    return code, stdout.getvalue(), stderr.getvalue()


def job_block(workflow: str, job: str) -> str | None:
    """The text of one top-level job, or None when the workflow lacks it."""
    start = workflow.find(f"\n  {job}:\n")
    if start < 0:
        return None
    following = re.search(r"\n  [A-Za-z0-9_-]+:\n", workflow[start + 1 :])
    return workflow[start : start + 1 + following.start()] if following else workflow[start:]


def assert_releases_reach_the_publisher(release: str, nix: str) -> None:
    """GitHub starts no workflow for a release that GITHUB_TOKEN publishes.

    release.yml therefore dispatches nix-packages.yml on the release tag, and
    only a run on a release tag may push to the cache.
    """
    dispatch = job_block(release, "dispatch-nix-packages")
    if dispatch is None:
        raise AssertionError("release.yml must dispatch nix-packages.yml after it publishes a release")
    rules = {
        "    needs: [publish-release]\n": "must run after publish-release",
        "    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')\n": "must run only for a tag push",
        "    permissions:\n      actions: write\n    steps:\n": "must hold only the actions: write permission",
        "GH_TOKEN: ${{ github.token }}": "must dispatch with GITHUB_TOKEN",
        'run: gh workflow run nix-packages.yml --repo "$GITHUB_REPOSITORY" --ref "$GITHUB_REF_NAME"': "must dispatch nix-packages.yml on the release tag",
    }
    for text, rule in rules.items():
        if text not in dispatch:
            raise AssertionError(f"the dispatch job {rule}")
    publish_steps = [step for step in nix.split("\n      - ") if PUBLISH_RUN in step]
    if len(publish_steps) != 2:
        raise AssertionError("each native Nix job must have one publish step")
    for step in publish_steps:
        if f"\n        {TAG_GATE}\n" not in step:
            raise AssertionError("a publish step must run only on a release tag")


class PublishNixCacheTests(unittest.TestCase):
    def test_pushes_the_three_package_outputs(self) -> None:
        nix = FakeNix()
        code, stdout, _ = run_main(nix)
        self.assertEqual(code, 0)
        self.assertIn("pushed the runtime closure of 3 aarch64-darwin output(s)", stdout)
        build = next(argv for argv, _ in nix.calls if argv[1] == "build")
        self.assertEqual(
            build[-3:],
            [f".#packages.{SYSTEM}.chelis", f".#packages.{SYSTEM}.chelis-runtime", f".#packages.{SYSTEM}.chelisup"],
        )
        [(argv, stdin)] = nix.pushes()
        self.assertEqual(
            argv,
            ["nix", "run", "--inputs-from", ".", "nixpkgs#cachix", "--", "push", publish_nix_cache.CACHE_NAME],
        )
        self.assertEqual(stdin, "".join(f"{path}\n" for path in sorted([CHELIS, RUNTIME, CHELISUP])))

    def test_missing_token_runs_no_command(self) -> None:
        nix = FakeNix()
        code, _, stderr = run_main(nix, environ={})
        self.assertEqual(code, 2)
        self.assertIn(publish_nix_cache.TOKEN_VARIABLE, stderr)
        self.assertEqual(nix.calls, [])

    def test_failed_build_stops_before_the_push(self) -> None:
        nix = FakeNix(fail="build")
        code, _, stderr = run_main(nix)
        self.assertEqual(code, 1)
        self.assertIn("build failed", stderr)
        self.assertEqual(nix.pushes(), [])

    def test_failed_push_fails_the_release_job(self) -> None:
        nix = FakeNix(fail="run")
        code, stdout, stderr = run_main(nix)
        self.assertEqual(code, 1)
        self.assertIn("run failed", stderr)
        self.assertNotIn("pushed", stdout)

    def test_dry_run_lists_the_outputs_without_a_token(self) -> None:
        nix = FakeNix()
        code, stdout, _ = run_main(nix, argv=["--dry-run"], environ={})
        self.assertEqual(code, 0)
        self.assertEqual(stdout.splitlines(), sorted([CHELIS, RUNTIME, CHELISUP]))
        self.assertEqual(nix.pushes(), [])


class ReleaseTriggerContractTests(unittest.TestCase):
    """A published release must start the run that pushes to the cache."""

    def workflows(self) -> dict[str, str]:
        return {
            "release": RELEASE_WORKFLOW.read_text(encoding="utf-8"),
            "nix": NIX_WORKFLOW.read_text(encoding="utf-8"),
        }

    def test_release_dispatches_the_publishing_nix_run(self) -> None:
        assert_releases_reach_the_publisher(**self.workflows())

    def test_each_trigger_rule_rejects_its_mutation(self) -> None:
        darwin_publish = (
            "        run: nix flake check --print-build-logs\n"
            "      - name: Publish the package closure to Cachix\n"
            f"        {TAG_GATE}\n"
        )
        mutations = [
            (
                "release",
                "\n  dispatch-nix-packages:\n",
                "\n  dispatch-nix-packages-retired:\n",
                "must dispatch nix-packages.yml after it publishes",
            ),
            ("release", "    needs: [publish-release]\n", "    needs: [preflight]\n", "must run after publish-release"),
            (
                "release",
                "    name: Dispatch the Nix package checks\n"
                "    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')\n",
                "    name: Dispatch the Nix package checks\n    if: github.event_name == 'workflow_dispatch'\n",
                "must run only for a tag push",
            ),
            (
                "release",
                "      actions: write\n    steps:\n",
                "      actions: write\n      contents: write\n    steps:\n",
                "only the actions: write permission",
            ),
            ("release", '--ref "$GITHUB_REF_NAME"', "--ref main", "on the release tag"),
            (
                "nix",
                f"      # any other ref pushes nothing.\n      - name: Publish the package closure to Cachix\n        {TAG_GATE}\n",
                "      # any other ref pushes nothing.\n      - name: Publish the package closure to Cachix\n"
                "        if: github.event_name == 'release'\n",
                "must run only on a release tag",
            ),
            (
                "nix",
                darwin_publish,
                "        run: nix flake check --print-build-logs\n      - name: Publish the package closure to Cachix\n",
                "must run only on a release tag",
            ),
            (
                "nix",
                darwin_publish
                + "        env:\n          CACHIX_AUTH_TOKEN: ${{ secrets.CACHIX_DEPLOY }}\n"
                + f"        {PUBLISH_RUN}\n",
                "        run: nix flake check --print-build-logs\n",
                "each native Nix job must have one publish step",
            ),
        ]
        for name, old, new, message in mutations:
            with self.subTest(workflow=name, message=message):
                workflows = self.workflows()
                self.assertEqual(workflows[name].count(old), 1, old)
                workflows[name] = workflows[name].replace(old, new)
                with self.assertRaisesRegex(AssertionError, re.escape(message)):
                    assert_releases_reach_the_publisher(**workflows)


class FlakeCacheContractTests(unittest.TestCase):
    """The flake trusts the cache that the release publisher pushes to."""

    def flake_list(self, attribute: str) -> list[str]:
        flake = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")
        match = re.search(rf"nixConfig\.{re.escape(attribute)}\s*=\s*\[(?P<body>[^\]]*)\]", flake)
        self.assertIsNotNone(match, f"flake.nix declares no nixConfig.{attribute}")
        assert match is not None
        return re.findall(r'"([^"]*)"', match.group("body"))

    def test_flake_substitutes_from_the_published_cache(self) -> None:
        self.assertEqual(
            self.flake_list("extra-substituters"),
            [f"https://{publish_nix_cache.CACHE_NAME}.cachix.org"],
        )

    def test_flake_trusts_a_valid_key_of_the_published_cache(self) -> None:
        [key] = self.flake_list("extra-trusted-public-keys")
        name, _, encoded = key.partition(":")
        self.assertEqual(name, f"{publish_nix_cache.CACHE_NAME}.cachix.org-1")
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except ValueError:
            decoded = b""
        self.assertEqual(
            len(decoded),
            32,
            "the trusted key must be the cache's Ed25519 public key, not a placeholder",
        )


if __name__ == "__main__":
    unittest.main()
