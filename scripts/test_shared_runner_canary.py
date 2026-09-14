"""Positive and negative contracts for docs/shared_runner_canary.md."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from scripts import shared_runner_canary as canary

SHA = "a" * 40
STORE_PATH = "/nix/store/" + "a" * 32 + "-shared-runner-probe"
NAR_HASH = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="


def environment():
    return {
        "GITHUB_ACTIONS": "true",
        "RUNNER_ENVIRONMENT": "self-hosted",
        "GITHUB_REPOSITORY": "Chelis-Lang/chelis",
        "GITHUB_REPOSITORY_ID": "1201277270",
        "GITHUB_REPOSITORY_OWNER_ID": "275816231",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_SHA": SHA,
        "GITHUB_WORKFLOW_SHA": SHA,
        "GITHUB_WORKFLOW_REF": "Chelis-Lang/chelis/.github/workflows/shared-runner-canary.yml@refs/heads/main",
        "GITHUB_RUN_ID": "123456",
        "GITHUB_RUN_ATTEMPT": "1",
        "RUNNER_TEMP": "/tmp/fixture",
    }


class CanaryTests(unittest.TestCase):
    def test_manual_main_context_is_parsed_once(self):
        job = canary.Job.parse(environment())
        self.assertEqual((job.run_id, job.attempt, job.sha), ("123456", "1", SHA))
        self.assertEqual(job.temp, Path("/tmp/fixture"))

    def test_wrong_context_fails_without_a_successful_skip(self):
        for name, values in {
            "GITHUB_ACTIONS": ["false", ""],
            "RUNNER_ENVIRONMENT": ["github-hosted", ""],
            "GITHUB_REPOSITORY": ["Chelis-Lang/ci", "other/chelis"],
            "GITHUB_REPOSITORY_ID": ["1"],
            "GITHUB_REPOSITORY_OWNER_ID": ["1"],
            "GITHUB_EVENT_NAME": [
                "push",
                "pull_request",
                "pull_request_target",
                "schedule",
            ],
            "GITHUB_REF": ["refs/pull/1/merge", "refs/heads/topic"],
            "GITHUB_SHA": ["0" * 40, "not-a-sha"],
            "GITHUB_WORKFLOW_SHA": ["b" * 40],
            "GITHUB_WORKFLOW_REF": ["other/workflow@refs/heads/main"],
            "GITHUB_RUN_ID": ["0", "01", "$(id)"],
            "GITHUB_RUN_ATTEMPT": ["0", "01"],
            "RUNNER_TEMP": ["relative", ""],
        }.items():
            for value in values:
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(canary.ProbeError),
                ):
                    canary.Job.parse(environment() | {name: value})
        with (
            patch.dict(canary.os.environ, {}, clear=True),
            patch.object(canary.sys, "argv", ["shared_runner_canary.py"]),
            patch("scripts.shared_runner_canary.probe") as probe,
        ):
            self.assertNotEqual(canary.main(), 0)
            probe.assert_not_called()

    def test_child_environment_cannot_inherit_credentials(self):
        with tempfile.TemporaryDirectory() as root:
            work = Path(root)
            with patch.dict(
                canary.os.environ,
                {
                    "AWS_ACCESS_KEY_ID": "sentinel",
                    "CI_APP_PRIVATE_KEY": "sentinel",
                    "GH_TOKEN": "sentinel",
                    "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "sentinel",
                    "NIX_CONFIG": "require-sigs = false",
                },
            ):
                env = canary.child_environment(work)
            self.assertNotIn("sentinel", env.values())
            self.assertNotIn("NIX_CONFIG", env)
            self.assertNotIn("AWS_PROFILE", env)
            self.assertEqual(env["AWS_CONFIG_FILE"], "/dev/null")
            self.assertEqual(env["KACHE_S3_ACCESS_KEY"], "tunnet-member")
            self.assertEqual(env["KACHE_S3_ENDPOINT"], "https://kache.mesh.cproof.ai")

    def test_nar_record_requires_valid_hash_and_exact_store_path(self):
        info = {
            STORE_PATH: {
                "narHash": NAR_HASH,
                "signatures": [canary.SIGNING_KEY + ":fixture"],
            }
        }
        self.assertEqual(
            canary.NarRecord.parse(json.dumps(info), STORE_PATH).nar_hash, NAR_HASH
        )
        for value in (
            {},
            {"/tmp/probe": info[STORE_PATH]},
            {STORE_PATH: {"narHash": "sha256-bad", "signatures": []}},
            {STORE_PATH: {"narHash": NAR_HASH, "signatures": "wrong"}},
        ):
            with self.assertRaises(canary.ProbeError):
                canary.NarRecord.parse(json.dumps(value), STORE_PATH)

    def test_readback_requires_hash_parity_and_reviewed_signature(self):
        before = canary.NarRecord(NAR_HASH, ())
        after = canary.NarRecord(NAR_HASH, (canary.SIGNING_KEY + ":fixture",))
        canary.require_readback(before, after)
        for invalid in (
            canary.NarRecord(NAR_HASH, ()),
            canary.NarRecord(
                "sha256-AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=", after.signatures
            ),
            canary.NarRecord(NAR_HASH, ("other:fixture",)),
        ):
            with self.assertRaises(canary.ProbeError):
                canary.require_readback(before, invalid)

    def test_entry_inventory_is_nonempty_and_detects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            cache = Path(root)
            entry = cache / "store" / ("b" * 64)
            entry.mkdir(parents=True)
            artifact = entry / "artifact.rlib"
            artifact.write_bytes(b"one")
            before = canary.entry_digests(cache)
            self.assertEqual(len(before), 1)
            artifact.write_bytes(b"two")
            self.assertNotEqual(canary.entry_digests(cache), before)
            artifact.unlink()
            with self.assertRaises(canary.ProbeError):
                canary.entry_digests(cache)
            artifact.symlink_to("/etc/passwd")
            with self.assertRaises(canary.ProbeError):
                canary.entry_digests(cache)

    def test_actual_crate_hit_is_required(self):
        report = {
            "summary": {"local_hits": 1, "errors": 0, "store_failures": 0},
            "traceEvents": [{"name": "hit: fixture"}],
        }
        self.assertEqual(canary.local_hits(json.dumps(report), "fixture"), 1)
        for changes in (
            {"summary": {"local_hits": 0, "errors": 0, "store_failures": 0}},
            {"summary": {"local_hits": 1, "errors": 1, "store_failures": 0}},
            {"summary": {"local_hits": 1, "errors": False, "store_failures": 0}},
            {"summary": {"local_hits": True, "errors": 0, "store_failures": 0}},
            {"traceEvents": [{"name": "hit: other"}]},
            {"traceEvents": []},
        ):
            with self.assertRaises(canary.ProbeError):
                canary.local_hits(json.dumps(report | changes), "fixture")

    def test_probe_expression_uses_store_python_not_a_shell(self):
        expression = canary.probe_expression(
            "123-1-abc", "/nix/store/" + "b" * 32 + "-python3-3.11/bin/python3"
        )
        self.assertIn('system = "x86_64-linux"', expression)
        self.assertIn('"-I" "-S" "-c"', expression)
        self.assertNotIn("/bin/sh", expression)
        for identity, python in (("$(id)", "/bin/python3"), ("123-1", "/tmp/python3")):
            with self.assertRaises(canary.ProbeError):
                canary.probe_expression(identity, python)

    def test_host_identity_requires_the_existing_ec2_and_reviewed_system_shape(self):
        system = (
            "/nix/store/" + "a" * 32 + "-nixos-system-chelis-ci-warm-amazon-fixture"
        )
        self.assertEqual(
            canary.Host.parse("i-0f07f7850a4551b01", system).system, system
        )
        for instance, path in (
            ("i-0c39d151b53a7698e", system),
            ("i-0f07f7850a4551b01", "/tmp/system"),
        ):
            with self.assertRaises(canary.ProbeError):
                canary.Host.parse(instance, path)

    def test_nix_orchestration_requires_tokenless_push_and_checked_readback(self):
        before = json.dumps({STORE_PATH: {"narHash": NAR_HASH, "signatures": []}})
        signed = json.dumps(
            {
                STORE_PATH: {
                    "narHash": NAR_HASH,
                    "signatures": [canary.SIGNING_KEY + ":fixture"],
                }
            }
        )
        for after, accepted in ((signed, True), (before, False)):
            with (
                tempfile.TemporaryDirectory() as directory,
                patch.object(canary, "probe_expression", return_value="fixture"),
                patch.object(
                    canary, "run", side_effect=[STORE_PATH, before, "", "", after]
                ) as run,
            ):
                work = Path(directory)
                if accepted:
                    receipt = canary.nix_probe(
                        "123-1-abc", canary.child_environment(work), work
                    )
                    self.assertTrue(receipt["readback_hash_identical"])
                    push = run.call_args_list[2].args[0]
                    self.assertIn("--tunnet", push)
                    self.assertNotIn("--auth-token-script", push)
                    copied = run.call_args_list[3].args[0]
                    self.assertIn(canary.PUBLIC_KEY, copied)
                    self.assertNotIn("--no-check-sigs", copied)
                    self.assertIn("require-sigs", copied)
                else:
                    with self.assertRaisesRegex(canary.ProbeError, "signature"):
                        canary.nix_probe(
                            "123-1-abc", canary.child_environment(work), work
                        )

    def test_kache_orchestration_clears_local_state_and_rejects_altered_readback(self):
        for alter in (False, True):
            with self.subTest(alter=alter), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                key = "a" * 64
                operations = []

                def run(argv, env, _cwd, operation, **_kwargs):
                    operations.append(operation)
                    entry = work / "cache" / "store" / key
                    if operation == "first Rust build":
                        entry.mkdir(parents=True)
                        (entry / "artifact.rlib").write_bytes(b"fixture")
                        (work / "cache/index.db").write_bytes(b"fixture-index")
                        (work / "target").mkdir()
                    elif operation == "Kache readback":
                        self.assertEqual(list((work / "cache").iterdir()), [])
                        self.assertFalse((work / "target").exists())
                        self.assertIn("--workspace", argv)
                        entry.mkdir(parents=True)
                        (entry / "artifact.rlib").write_bytes(
                            b"altered" if alter else b"fixture"
                        )
                    elif operation == "second Rust build":
                        config = tomllib.loads((work / "kache.toml").read_text())
                        self.assertIs(config["cache"]["local_only"], True)
                        self.assertNotIn("AWS_PROFILE", env)
                    elif operation == "Kache report":
                        return json.dumps(
                            {
                                "summary": {
                                    "local_hits": 1,
                                    "errors": 0,
                                    "store_failures": 0,
                                },
                                "traceEvents": [
                                    {"name": "hit: shared_runner_123_1_abc"}
                                ],
                            }
                        )
                    return ""

                with (
                    patch.object(canary, "Daemon"),
                    patch.object(canary, "run", side_effect=run),
                ):
                    if alter:
                        with self.assertRaisesRegex(
                            canary.ProbeError, "restored cache bytes"
                        ):
                            canary.kache_probe(
                                "123-1-abc", canary.child_environment(work), work
                            )
                        self.assertNotIn("second Rust build", operations)
                    else:
                        receipt = canary.kache_probe(
                            "123-1-abc", canary.child_environment(work), work
                        )
                        self.assertEqual(receipt["local_hits"], 1)
                        self.assertEqual(
                            operations,
                            [
                                "first Rust build",
                                "Kache upload",
                                "Kache readback",
                                "second Rust build",
                                "Kache report",
                            ],
                        )

    def test_failed_command_does_not_disclose_subprocess_output(self):
        with patch.object(
            canary.subprocess,
            "run",
            return_value=canary.subprocess.CompletedProcess(
                [], 1, "credential-sentinel", "credential-sentinel"
            ),
        ):
            with self.assertRaises(canary.ProbeError) as error:
                canary.run(["fixture"], {}, Path("/tmp"), "fixture")
            self.assertNotIn("credential-sentinel", str(error.exception))

    def test_workflow_is_manual_pinned_and_has_no_app_or_oidc_secret(self):
        root = Path(__file__).resolve().parents[1]
        workflow_path = ".github/workflows/shared-runner-canary.yml"
        workflow = (root / workflow_path).read_text()
        self.assertEqual(
            (root / ".github/actionlint.yaml").read_text(),
            "self-hosted-runner:\n  labels:\n    - chelis-ci-warm-x64\n",
        )
        rules = tomllib.loads((root / ".config/ci-test-targets.toml").read_text())[
            "path_rule"
        ]
        for path in (
            ".github/actionlint.yaml",
            workflow_path,
            "scripts/shared_runner_canary.py",
            "scripts/test_shared_runner_canary.py",
        ):
            matches = [rule for rule in rules if rule["prefix"] == path]
            self.assertEqual(len(matches), 1, path)
            self.assertEqual(matches[0]["disposition"], "owner")
            self.assertEqual(matches[0]["workflow"], "ci.yml")
            self.assertEqual(matches[0]["job"], "script-unit")
        self.assertIn("on:\n  workflow_dispatch:", workflow)
        self.assertIn("group: chelis-ci-trusted", workflow)
        self.assertIn("labels: chelis-ci-warm-x64", workflow)
        self.assertIn("persist-credentials: false", workflow)
        self.assertIn("github.event.repository.private", workflow)
        self.assertIn(".venv/bin/python scripts/shared_runner_canary.py", workflow)
        for denied in (
            "secrets.",
            "id-token:",
            "pull_request:",
            "schedule:",
            "sudo ",
            "@main",
            "@v6",
            "secrets: inherit",
        ):
            self.assertNotIn(denied, workflow)


if __name__ == "__main__":
    unittest.main()
